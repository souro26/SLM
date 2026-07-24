"""
model/transformer.py

Full decoder-only transformer stack: token embedding -> shared RoPE table
-> n_layer TransformerBlocks -> final RMSNorm -> tied unembedding.

Weight tying: there is no separate output projection matrix. The final
logits are computed as F.linear(x, self.token_emb.weight) — literally the
same tensor used for the input embedding, not a copy kept in sync. This is
what "tied" means; a separate nn.Linear head that gets its weight
reassigned post-hoc risks silently drifting or being double-initialized.

Initialization (applied here, once, after construction):
    - All Linear/Embedding weights: normal(mean=0, std=0.02)
    - Every Linear tagged SCALE_RESIDUAL_INIT (attention's o_proj, ffn's
      down_proj) is instead initialized at std=0.02 / sqrt(2 * n_layer) —
      these are writes back into the residual stream, and starting them
      small keeps the residual stream well-scaled early in training.
    - Biases: none anywhere in this model (see config), nothing to init.

KV caching: forward() takes and returns an explicit list[KVCache | None],
one slot per layer — the model holds no cache state itself. This mirrors
how TransformerBlock/GQAAttention already return (output, new_cache)
rather than mutating internal state, and it means one model instance can
safely serve multiple concurrent generation sequences without cache
collisions.

Usage:
    from model.transformer import TransformerModel

    model = TransformerModel(cfg)
    logits, kv_caches = model(input_ids)                       # prefill
    logits, kv_caches = model(next_token_ids, kv_caches=kv_caches)  # decode step
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as f
from torch import nn

from model.attention import KVCache
from model.block import TransformerBlock
from model.config import ModelConfig
from model.rmsnorm import RMSNorm
from model.rope import RotaryEmbedding

logger = logging.getLogger(__name__)

_INIT_STD = 0.02
_TARGET_PARAM_COUNT = 85_600_000
_PARAM_COUNT_WARN_TOLERANCE = 0.05


class TransformerModel(nn.Module):
    """Full decoder-only transformer: embedding, n_layer blocks, tied unembedding."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.context_length)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model)

        self._init_weights()
        self._log_param_count()

    def _init_weights(self) -> None:
        residual_std = _INIT_STD / math.sqrt(2 * self.cfg.n_layer)
        n_base, n_scaled = 0, 0

        for module in self.modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "SCALE_RESIDUAL_INIT", False):
                    nn.init.normal_(module.weight, mean=0.0, std=residual_std)
                    n_scaled += 1
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=_INIT_STD)
                    n_base += 1
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=_INIT_STD)

        logger.info(
            "weight init: %d base Linear layers (std=%.4f), "
            "%d residual-scaled Linear layers (std=%.5f)",
            n_base,
            _INIT_STD,
            n_scaled,
            residual_std,
        )

    def _log_param_count(self) -> None:
        n_params = self.count_parameters()
        deviation = abs(n_params - _TARGET_PARAM_COUNT) / _TARGET_PARAM_COUNT
        logger.info(
            "model constructed: n_layer=%d, d_model=%d, %d total parameters (%.2fM)",
            self.cfg.n_layer,
            self.cfg.d_model,
            n_params,
            n_params / 1e6,
        )
        if deviation > _PARAM_COUNT_WARN_TOLERANCE:
            logger.warning(
                "parameter count deviates from ~%.0fM target by %.1f%% (%d params)",
                _TARGET_PARAM_COUNT / 1e6,
                deviation * 100,
                n_params,
            )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: list[KVCache | None] | None = None,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        b, t = input_ids.shape

        if kv_caches is None:
            kv_caches = [None] * self.cfg.n_layer
        elif len(kv_caches) != self.cfg.n_layer:
            raise ValueError(
                f"kv_caches has {len(kv_caches)} entries, expected one per "
                f"layer ({self.cfg.n_layer})"
            )

        start_pos = kv_caches[0].seq_len if kv_caches[0] is not None else 0
        cos, sin = self.rope(seq_len=t, device=input_ids.device, start_pos=start_pos)

        x = self.token_emb(input_ids)

        new_caches: list[KVCache] = []

        def _create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        for block, cache in zip(self.blocks, kv_caches, strict=False):
            if getattr(self, "gradient_checkpointing", False) and self.training:
                import torch.utils.checkpoint

                x, new_cache = torch.utils.checkpoint.checkpoint(
                    _create_custom_forward(block),
                    x,
                    cos,
                    sin,
                    cache,
                    use_reentrant=False,
                )
            else:
                x, new_cache = block(x, cos, sin, kv_cache=cache)

            new_caches.append(new_cache)

        x = self.final_norm(x)
        logits = f.linear(x, self.token_emb.weight)
        return logits, new_caches
