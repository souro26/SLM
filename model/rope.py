"""
model/rope.py

Rotary positional embeddings (RoPE). Rotates query and key vectors in
pairs of dimensions as a function of absolute position, so that the dot
product between a query at position m and a key at position n depends
only on their relative distance (m - n), not their absolute positions.

Applied identically to every attention head (query heads and KV heads
alike) — RoPE operates per-head on head_dim, independent of how many
heads there are, so it works unchanged under GQA.

Precompute once per model (cos/sin tables up to context_length), then
apply cheaply at every attention call — no extra parameters, nothing
learned here.

Usage:
    from model.rope import RotaryEmbedding, apply_rotary_pos_emb

    rope = RotaryEmbedding(head_dim=64, max_seq_len=2048, theta=10000.0)
    cos, sin = rope(seq_len=x.shape[1], device=x.device)  # [seq_len, head_dim]
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
"""

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin tables up to max_seq_len."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE pairing, got {head_dim}")

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        cos, sin = self._build_cache(max_seq_len)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _build_cache(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def forward(
        self, seq_len: int, device: torch.device, start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin for positions [start_pos, start_pos + seq_len)."""
        end_pos = start_pos + seq_len
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"requested positions up to {end_pos} exceed max_seq_len "
                f"({self.max_seq_len}) the RoPE cache was built for"
            )
        cos = self.cos_cached[start_pos:end_pos]
        sin = self.sin_cached[start_pos:end_pos]
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dim in half and rotate: [x1, x2] -> [-x2, x1]."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE rotation to query and key tensors."""
    cos = cos.unsqueeze(0).unsqueeze(0).to(q.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(q.dtype)
    q_rotated = (q * cos) + (_rotate_half(q) * sin)
    k_rotated = (k * cos) + (_rotate_half(k) * sin)
    return q_rotated, k_rotated
