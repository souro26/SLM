"""
train/logger.py

Abstracts training metrics logging.
Supports Weights & Biases (wandb) and local CSV logging.

Train-step metrics and eval metrics have different key sets (e.g. "loss"/
"lr_muon"/"grad_norm" vs "val_loss"/"perplexity"). A single CSV file can't
sensibly hold both — the header is only ever written once, so if the first
call's keys differ from a later call's keys, every row after the first
mismatched call has a different column set than the header describes,
producing a CSV that misaligns when opened in pandas/Excel. To avoid this,
log_metrics takes a `tag` (e.g. "train", "eval") and routes each tag to its
own CSV file — one stable header per file, for the file's entire lifetime.

Usage:
    from train.logger import TrainLogger

    train_logger = TrainLogger(cfg)
    train_logger.log_metrics(step, {"loss": 2.5, "lr": 3e-4}, tag="train")
    train_logger.log_metrics(step, {"val_loss": 2.1}, tag="eval")
    train_logger.close()
"""

from __future__ import annotations

import csv
import dataclasses
import logging
from pathlib import Path
from typing import Any

import torch

import wandb
from train.config import TrainConfig

logger = logging.getLogger(__name__)


def _coerce_scalar(v: Any) -> Any:
    """Convert torch scalar tensors to plain Python numbers before logging."""
    if torch.is_tensor(v):
        if v.numel() != 1:
            raise ValueError(
                f"log_metrics received a non-scalar tensor (shape {tuple(v.shape)}) "
                "— pass .item() or a reduced scalar instead"
            )
        return v.item()
    return v


class TrainLogger:
    """Manages routing of metrics to stdout, CSV, and wandb."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.use_wandb = cfg.logging.wandb.enabled
        self.use_csv = cfg.logging.csv.enabled

        self._csv_base_path = None
        self._csv_headers_written: dict[str, bool] = {}
        self._csv_fieldnames: dict[str, list[str]] = {}

        if self.use_wandb:
            try:
                import wandb

                wandb.init(
                    project=cfg.logging.wandb.project,
                    entity=cfg.logging.wandb.entity,
                    name=cfg.run_name,
                    config=dataclasses.asdict(cfg),
                )
            except ImportError:
                logger.warning(
                    "wandb is enabled in config but not installed. Disabling wandb logging."
                )
                self.use_wandb = False

        if self.use_csv and cfg.logging.csv.path:
            self._csv_base_path = Path(cfg.logging.csv.path)
            self._csv_base_path.parent.mkdir(parents=True, exist_ok=True)

    def _csv_path_for_tag(self, tag: str) -> Path:
        """e.g. logs/pilot-001.csv + tag "eval" -> logs/pilot-001.eval.csv"""
        return self._csv_base_path.with_suffix(f".{tag}{self._csv_base_path.suffix}")

    def log_metrics(self, step: int, metrics: dict[str, Any], tag: str = "train") -> None:
        """Log a dictionary of metrics for a given step to console, wandb, and CSV."""
        metrics = {k: _coerce_scalar(v) for k, v in metrics.items()}

        parts = [f"[{tag}] Step {step:06d}"]
        if "loss" in metrics:
            parts.append(f"Loss {metrics['loss']:.4f}")

        for k, v in metrics.items():
            if k == "loss":
                continue
            if isinstance(v, float):
                if 0 < abs(v) < 1e-3 or abs(v) > 1e4:
                    parts.append(f"{k} {v:.2e}")
                else:
                    parts.append(f"{k} {v:.4f}")
            else:
                parts.append(f"{k} {v}")

        logger.info(" | ".join(parts))

        if self.use_wandb:
            wandb.log({f"{tag}/{k}": v for k, v in metrics.items()}, step=step)

        if self.use_csv and self._csv_base_path is not None:
            self._write_csv_row(tag, step, metrics)

    def _write_csv_row(self, tag: str, step: int, metrics: dict[str, Any]) -> None:
        csv_path = self._csv_path_for_tag(tag)
        row = {"step": step, **metrics}

        if tag not in self._csv_fieldnames:
            self._csv_fieldnames[tag] = list(row.keys())
            self._csv_headers_written[tag] = csv_path.exists()

        fieldnames = self._csv_fieldnames[tag]
        if list(row.keys()) != fieldnames:
            logger.error(
                "CSV row for tag '%s' has keys %s but the established schema is %s "
                "— dropping extra/missing keys to keep the file well-formed",
                tag,
                list(row.keys()),
                fieldnames,
            )
            row = {k: row.get(k, "") for k in fieldnames}

        with open(csv_path, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not self._csv_headers_written[tag]:
                writer.writeheader()
                self._csv_headers_written[tag] = True
            writer.writerow(row)

    def close(self) -> None:
        """Flush and close all logging streams."""
        if self.use_wandb:
            import wandb

            wandb.finish()
