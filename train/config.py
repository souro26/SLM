"""
train/config.py

Loads and validates a training run config (configs/train_pilot.yaml or
configs/train_full.yaml) into typed, nested dataclasses. Mirrors the
pattern in model/config.py — everything downstream imports TrainConfig
from here rather than reading YAML directly.

The YAML has real nested sections (optimizer.muon, optimizer.adamw,
lr_schedule, logging.wandb, logging.csv) — each gets its own small
dataclass rather than flattening everything into one giant TrainConfig,
so a bug in e.g. Muon's fields can't be confused with a bug in the LR
schedule's fields.

Usage:
    from train.config import TrainConfig

    cfg = TrainConfig.from_yaml("configs/train_pilot.yaml")
    print(cfg.optimizer.muon.lr, cfg.lr_schedule.warmup_steps)
    print(cfg.effective_batch_size)
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_SUPPORTED_DECAY_SCHEDULES = {"cosine"}


@dataclasses.dataclass
class MuonConfig:
    lr: float
    weight_decay: float
    momentum: float
    nesterov: bool
    ns_steps: int


@dataclasses.dataclass
class AdamWConfig:
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float


@dataclasses.dataclass
class OptimizerConfig:
    muon: MuonConfig
    adamw: AdamWConfig


@dataclasses.dataclass
class LRScheduleConfig:
    warmup_steps: int
    decay: str
    min_lr_ratio: float


@dataclasses.dataclass
class WandbConfig:
    enabled: bool
    project: str
    entity: str | None


@dataclasses.dataclass
class CSVConfig:
    enabled: bool
    path: str


@dataclasses.dataclass
class LoggingConfig:
    wandb: WandbConfig
    csv: CSVConfig


@dataclasses.dataclass
class TrainConfig:
    run_name: str
    seed: int

    train_data: str
    val_data: str
    tokenizer_path: str

    max_tokens: int

    micro_batch_size: int
    grad_accum_steps: int

    optimizer: OptimizerConfig
    lr_schedule: LRScheduleConfig

    grad_clip: float

    gradient_checkpointing: bool
    compile: bool

    checkpoint_dir: str
    save_every_steps: int
    keep_last_n: int
    resume_from: str | None

    log_every_steps: int
    eval_every_steps: int
    logging: LoggingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load and validate a training config from a YAML file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        muon_raw = raw["optimizer"]["muon"]
        adamw_raw = raw["optimizer"]["adamw"]
        optimizer = OptimizerConfig(
            muon=MuonConfig(**muon_raw),
            adamw=AdamWConfig(
                lr=adamw_raw["lr"],
                betas=tuple(adamw_raw["betas"]),
                eps=adamw_raw["eps"],
                weight_decay=adamw_raw["weight_decay"],
            ),
        )

        lr_schedule = LRScheduleConfig(**raw["lr_schedule"])

        wandb_raw = raw["logging"]["wandb"]
        csv_raw = raw["logging"]["csv"]
        logging_cfg = LoggingConfig(
            wandb=WandbConfig(**wandb_raw),
            csv=CSVConfig(**csv_raw),
        )

        cfg = cls(
            run_name=raw["run_name"],
            seed=raw["seed"],
            train_data=raw["train_data"],
            val_data=raw["val_data"],
            tokenizer_path=raw["tokenizer_path"],
            max_tokens=raw["max_tokens"],
            micro_batch_size=raw["micro_batch_size"],
            grad_accum_steps=raw["grad_accum_steps"],
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            grad_clip=raw["grad_clip"],
            gradient_checkpointing=raw["gradient_checkpointing"],
            compile=raw["compile"],
            checkpoint_dir=raw["checkpoint_dir"],
            save_every_steps=raw["save_every_steps"],
            keep_last_n=raw["keep_last_n"],
            resume_from=raw.get("resume_from"),
            log_every_steps=raw["log_every_steps"],
            eval_every_steps=raw["eval_every_steps"],
            logging=logging_cfg,
        )
        cfg._validate(path)
        return cfg

    def _validate(self, source_path: Path) -> None:
        """Cross-check fields and warn/raise on missing paths."""
        if self.micro_batch_size <= 0:
            raise ValueError(f"micro_batch_size must be > 0, got {self.micro_batch_size}")
        if self.grad_accum_steps <= 0:
            raise ValueError(f"grad_accum_steps must be > 0, got {self.grad_accum_steps}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")
        if self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be > 0, got {self.grad_clip}")

        if self.lr_schedule.decay not in _SUPPORTED_DECAY_SCHEDULES:
            raise ValueError(
                f"lr_schedule.decay='{self.lr_schedule.decay}' is not supported "
                f"— expected one of {sorted(_SUPPORTED_DECAY_SCHEDULES)}"
            )
        if self.lr_schedule.warmup_steps < 0:
            raise ValueError(
                f"lr_schedule.warmup_steps must be >= 0, got {self.lr_schedule.warmup_steps}"
            )
        if not (0.0 < self.lr_schedule.min_lr_ratio <= 1.0):
            raise ValueError(
                f"lr_schedule.min_lr_ratio must be in (0, 1], got {self.lr_schedule.min_lr_ratio}"
            )

        if self.save_every_steps <= 0:
            raise ValueError(f"save_every_steps must be > 0, got {self.save_every_steps}")
        if self.eval_every_steps <= 0:
            raise ValueError(f"eval_every_steps must be > 0, got {self.eval_every_steps}")
        if self.log_every_steps <= 0:
            raise ValueError(f"log_every_steps must be > 0, got {self.log_every_steps}")
        if self.keep_last_n < 1:
            raise ValueError(f"keep_last_n must be >= 1, got {self.keep_last_n}")
        if self.resume_from is not None and not Path(self.resume_from).exists():
            raise ValueError(
                f"resume_from='{self.resume_from}' does not exist — "
                "refusing to silently start a fresh run instead of resuming"
            )

        log = logging.getLogger(__name__)
        for field_name in ("train_data", "val_data", "tokenizer_path"):
            value = getattr(self, field_name)
            if not Path(value).exists():
                log.warning(
                    "%s: %s='%s' does not exist relative to cwd — "
                    "make sure to run from the repo root, or that it's been generated yet",
                    source_path,
                    field_name,
                    value,
                )

    @property
    def effective_batch_size(self) -> int:
        """Sequences per optimizer step, after gradient accumulation."""
        return self.micro_batch_size * self.grad_accum_steps

    def tokens_per_step(self, context_length: int) -> int:
        """Tokens consumed per optimizer step."""
        return self.effective_batch_size * context_length

    def estimated_total_steps(self, context_length: int) -> int:
        """Rough total optimizer steps to reach max_tokens."""
        return self.max_tokens // self.tokens_per_step(context_length)
