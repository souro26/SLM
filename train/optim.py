"""
train/optim.py

Configures and instantiates the optimizers for the training loop.
Implements the Muon optimizer (Momentum Orthogonalized) for 2D parameters (nn.Linear)
using Newton-Schulz iteration in bfloat16, and PyTorch's AdamW for all 1D parameters
(RMSNorm, biases) and the tied embedding matrix.

Usage:
    from train.optim import create_optimizers
    muon_opt, adamw_opt = create_optimizers(model, cfg)
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
from torch import nn, optim

from train.config import TrainConfig

logger = logging.getLogger(__name__)


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. Runs in bfloat16 to maximize throughput on modern GPUs."""
    assert len(g.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()

    if x.size(0) > x.size(1):
        x = x.T

    x = x / (x.norm() + eps)
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x

    if g.size(0) > g.size(1):
        x = x.T

    return x.to(g.dtype)


class Muon(optim.Optimizer):
    """Muon optimizer."""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                # Momentum on the RAW gradient — buf.lerp_(g, 1 - momentum)
                # is momentum*buf + (1 - momentum)*g.
                buf.lerp_(g, 1.0 - momentum)

                # Nesterov look-ahead blend, still on the raw (non-orthogonalized) tensor.
                g = g.lerp(buf, momentum) if nesterov else buf

                # Orthogonalize ONCE, after momentum — not before. See class docstring.
                g_ortho = zeropower_via_newtonschulz5(g, steps=ns_steps)

                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5

                if weight_decay > 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                p.add_(g_ortho, alpha=-lr * scale)

        return loss


def create_optimizers(
    model: nn.Module, cfg: TrainConfig
) -> tuple[optim.Optimizer, optim.Optimizer]:
    """Routes 2D non-embedding parameters to Muon and everything else to AdamW."""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim >= 2 and not name.endswith("token_emb.weight"):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    muon_opt = Muon(
        muon_params,
        lr=cfg.optimizer.muon.lr,
        momentum=cfg.optimizer.muon.momentum,
        nesterov=cfg.optimizer.muon.nesterov,
        weight_decay=cfg.optimizer.muon.weight_decay,
        ns_steps=cfg.optimizer.muon.ns_steps,
    )

    adamw_opt = optim.AdamW(
        adamw_params,
        lr=cfg.optimizer.adamw.lr,
        betas=cfg.optimizer.adamw.betas,
        eps=cfg.optimizer.adamw.eps,
        weight_decay=cfg.optimizer.adamw.weight_decay,
    )

    logger.info("Optimizers configured:")
    logger.info("  Muon:  %d parameters (2D weights)", sum(p.numel() for p in muon_params))
    logger.info(
        "  AdamW: %d parameters (1D weights & embeddings)", sum(p.numel() for p in adamw_params)
    )

    return muon_opt, adamw_opt
