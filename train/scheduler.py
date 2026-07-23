"""
train/scheduler.py

Learning rate schedule: linear warmup from 0, then cosine decay down to
min_lr_ratio * peak_lr. Peak LR and min_lr_ratio come from TrainConfig
(configs/train_pilot.yaml's lr_schedule section + each optimizer's own
lr field).

Two optimizers (Muon, AdamW) each have their own peak LR in the config —
currently both 3e-4, but not guaranteed to always match, so this computes
a scale factor (0 to 1) rather than an absolute LR, and the caller applies
it to whichever optimizer's own peak LR is relevant. This keeps one
scheduler correct for both optimizers even if their peak LRs diverge.

    step < warmup_steps:  scale = step / warmup_steps               (linear)
    step >= warmup_steps: scale = min_lr_ratio + (1 - min_lr_ratio)
                                   * 0.5 * (1 + cos(pi * progress))  (cosine)

    where progress = (step - warmup_steps) / (total_steps - warmup_steps),
    clamped to [0, 1] so the schedule holds flat at min_lr_ratio for any
    step beyond total_steps rather than extrapolating the cosine further.

Usage:
    from train.scheduler import WarmupCosineSchedule

    schedule = WarmupCosineSchedule(
        warmup_steps=cfg.lr_schedule.warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=cfg.lr_schedule.min_lr_ratio,
    )
    scale = schedule.get_scale(step)
    for group in muon_optimizer.param_groups:
        group["lr"] = cfg.optimizer.muon.lr * scale
    for group in adamw_optimizer.param_groups:
        group["lr"] = cfg.optimizer.adamw.lr * scale
"""

from __future__ import annotations

import math


class WarmupCosineSchedule:
    """Linear warmup then cosine decay, expressed as a 0-1 scale factor."""

    def __init__(self, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if not (0.0 <= min_lr_ratio <= 1.0):
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

        self.warmup_steps = warmup_steps
        self.total_steps = max(0, total_steps)
        self.min_lr_ratio = min_lr_ratio

    def get_scale(self, step: int) -> float:
        """Return the LR scale factor (0 to 1) for the given step."""
        if step < 0:
            step = 0

        if self.warmup_steps > 0 and step < self.warmup_steps:
            return step / self.warmup_steps

        decay_span = max(1, self.total_steps - self.warmup_steps)
        progress = (step - self.warmup_steps) / decay_span
        progress = min(max(progress, 0.0), 1.0)

        cosine_term = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_term

    def get_lr(self, step: int, peak_lr: float) -> float:
        """Convenience: scale directly applied to a given peak LR."""
        return peak_lr * self.get_scale(step)
