"""
model/attention.py

Grouped-query self-attention (GQA). 8 query heads, 4 KV heads (2:1 grouping)
by default per configs/model.yaml. Uses torch.nn.functional.scaled_dot_product_attention
(SDPA) for the fused attention kernel — SDPA has no native GQA support, so KV
heads are expanded to match the query head count via _repeat_kv() before the
call.

RoPE is applied to Q and K (via model.rope.apply_rotary_pos_emb) before the
KV cache is updated, so cached K vectors are already rotated for their
absolute position — a newly appended token only needs its own rotation, not
a re-rotation of the whole cache.

Three masking cases, all handled explicitly (SDPA's is_causal=True only
covers the first):

  1. Plain causal prefill — no cache yet, full sequence in one call.
     is_causal=True, no explicit mask needed.
  2. Single-token decode — cache present, exactly one new token.
     The new token must attend to everything already cached plus itself.
     Since there's nothing after it, no masking is needed at all.
  3. Cache-prefill with multiple new tokens — cache present, more than one
     new token in this call (e.g. re-prefilling a prompt after a cache
     reset, or batched speculative tokens). New tokens must fully attend
     to the cache but stay causal *among themselves* — an explicit
     boolean mask is built for this case.

o_proj is tagged with SCALE_RESIDUAL_INIT = True. It is NOT scaled here —
it needs n_layers, which only the top-level model config in transformer.py
knows. transformer.py walks the module tree after construction and rescales
every tensor tagged this way by 1 / sqrt(2 * n_layers).

Usage:
    from model.attention import GQAAttention, KVCache
    from model.rope import RotaryEmbedding

    attn = GQAAttention(cfg)
    rope = RotaryEmbedding(cfg.head_dim, cfg.context_length)

    # prefill
    cos, sin = rope(seq_len=x.shape[1], device=x.device, start_pos=0)
    out, cache = attn(x, cos, sin, kv_cache=None)

    # decode, one token at a time
    cos, sin = rope(seq_len=1, device=x.device, start_pos=cache.seq_len)
    out, cache = attn(next_token_embed, cos, sin, kv_cache=cache)
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as f
from torch import nn

from model.config import ModelConfig
from model.rope import apply_rotary_pos_emb


@dataclasses.dataclass
class KVCache:
    """Accumulated KV tensors for one attention layer."""

    k: torch.Tensor
    v: torch.Tensor

    @property
    def seq_length(self) -> int:
        return self.k.shape[2]


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match query heads for GQA."""
    if n_rep == 1:
        return x
    b, n_kv, t, hd = x.shape
    x = x[:, :, None, :, :].expand(b, n_kv, n_rep, t, hd)
    return x.reshape(b, n_kv * n_rep, t, hd)


def _build_prefill_mask(q_len: int, cache_len: int, device: torch.device) -> torch.Tensor:
    """Boolean attention mask for cache-prefill with multiple new tokens."""
    total_len = cache_len + q_len
    mask = torch.zeros(q_len, total_len, dtype=torch.bool, device=device)
    mask[:, :cache_len] = True
    mask[:, cache_len:] = torch.tril(torch.ones(q_len, q_len, dtype=torch.bool, device=device))
    return mask


class GQAAttention(nn.Module):
    """Grouped-query self-attention with RoPE and optional KV caching."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads_q = cfg.n_heads_q
        self.n_heads_kv = cfg.n_heads_kv
        self.head_dim = cfg.head_dim
        self.gqa_group_size = cfg.gqa_group_size
        self.dropout_p = cfg.dropout

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads_q * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads_q * cfg.head_dim, cfg.d_model, bias=False)

        self.o_proj.SCALE_RESIDUAL_INIT = True

        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads_q, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads_kv, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            k = torch.cat([kv_cache.k, k], dim=2)
            v = torch.cat([kv_cache.v, v], dim=2)

        new_cache = KVCache(k=k, v=v)

        k_rep = _repeat_kv(k, self.gqa_group_size)
        v_rep = _repeat_kv(v, self.gqa_group_size)

        total_len = k_rep.shape[2]
        cache_len = total_len - t
        dropout_p = self.dropout_p if self.training else 0.0

        if cache_len == 0:
            attn_out = f.scaled_dot_product_attention(
                q, k_rep, v_rep, is_causal=True, dropout_p=dropout_p
            )
        elif t == 1:
            attn_out = f.scaled_dot_product_attention(
                q, k_rep, v_rep, is_causal=False, dropout_p=dropout_p
            )
        else:
            mask = _build_prefill_mask(t, cache_len, device=x.device)
            attn_out = f.scaled_dot_product_attention(
                q, k_rep, v_rep, attn_mask=mask, dropout_p=dropout_p
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, self.n_heads_q * self.head_dim)
        out = self.o_proj(attn_out)
        out = self.resid_dropout(out)
        return out, new_cache
