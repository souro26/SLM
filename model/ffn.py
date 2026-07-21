"""
model/ffn.py

SwiGLU feed-forward network. Three matrices instead of the usual two:
gate_proj and up_proj run in parallel on the same input, are combined with
a multiplicative gate, then projected back down to d_model.

    output = down_proj( silu(gate_proj(x)) * up_proj(x) )

Multiplicative gating (silu(gate) * up) learns sparser, more structured
representations than a single matrix + additive activation. d_ffn is set
to 8/3 * d_model (not the usual 4x) in configs/model.yaml specifically to
compensate for the extra matrix — this keeps total FFN parameter count
equivalent to a standard 4x two-matrix FFN, so d_ffn should already be
correct in the config; this module does no rescaling of its own.

down_proj is tagged SCALE_RESIDUAL_INIT = True, same as attention's
o_proj — it's a write back into the residual stream, so it gets the
1 / sqrt(2 * n_layers) init scaling applied later in transformer.py.

Usage:
    from model.ffn import SwiGLUFFN

    ffn = SwiGLUFFN(cfg)
    y = ffn(x)  # x: [..., d_model] -> y: [..., d_model]
"""

import torch
import torch.nn.functional as f
from torch import nn

from model.config import ModelConfig


class SwiGLUFFN(nn.Module):
    """SwiGLU feed forward loop."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ffn, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ffn, bias=False)
        self.down_proj = nn.Linear(cfg.d_ffn, cfg.d_model, bias=False)

        self.down_proj.SCALE_RESIDUAL_INIT = True
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = f.silu(self.gate_proj(x))
        up = self.up_proj(x)
        out = self.down_proj(gate * up)
        return self.dropout(out)
