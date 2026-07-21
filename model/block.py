"""
model/block.py

A single transformer decoder block: pre-norm attention sublayer followed by
pre-norm FFN sublayer, each wrapped in a residual connection.

    h = x + Attention(RMSNorm(x))
    out = h + FFN(RMSNorm(h))

Pre-norm (normalize before the sublayer, add its raw output to the
un-normed residual stream) rather than post-norm — gives cleaner gradient
flow through 24 stacked layers. The residual stream itself is never
normalized, only the copy fed into each sublayer.

Usage:
    from model.block import TransformerBlock

    block = TransformerBlock(cfg)
    out, kv_cache = block(x, cos, sin, kv_cache=None)
"""

import torch
from torch import nn

from model.attention import GQAAttention, KVCache
from model.config import ModelConfig
from model.ffn import SwiGLUFFN
from model.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """One decoder block, pre-norm attention + pre-norm FFN, residual around each."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = GQAAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        attn_out, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache=kv_cache)
        h = x + attn_out
        out = h + self.ffn(self.ffn_norm(h))
        return out, new_cache
