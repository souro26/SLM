"""
model/rmsnorm.py

RMSNorm is root-mean-square layer normalization, used pre-norm before every
attention and FFN sublayer in the transformer block.

Unlike LayerNorm, RMSNorm does NOT re-center (no mean subtraction) and has
no bias term. It only rescales activations by their root-mean-square, then
applies a learned per-channel scale. This is cheaper than LayerNorm and
empirically performs just as well in modern transformers.

    rms = sqrt(mean(x^2, dim=-1) + eps)
    output = (x / rms) * weight


Usage:
    from model.rmsnorm import RMSNorm

    norm = RMSNorm(dim=512)
    y = norm(x)  # x: [..., 512]
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Rescale x by its root-mean-square along the last dimension."""
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        out = self._normalize(x) * self.weight
        return out.to(input_dtype)
