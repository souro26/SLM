"""
train/logger.py

Abstracts training metrics logging.
Supports Weights & Biases (wandb) and local CSV logging.

Usage:
    from train.logger import TrainLogger

    train_logger = TrainLogger(cfg)
    train_logger.log_metrics(step, {"loss": 2.5, "lr": 3e-4})
    train_logger.close()
"""

from __future__ import annotations

import csv
import dataclasses
import logging
from pathlib import Path

from train.config import TrainConfig

logger = logging.getLogger(__name__)


class TrainLogger:
    """Manages routing of metrics to stdout, CSV, and wandb."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.use_wandb = cfg.logging.wandb.enabled
        self.use_csv = cfg.logging.csv.enabled

        self.csv_path = None
        self.csv_headers_written = False

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
            self.csv_path = Path(cfg.logging.csv.path)
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_headers_written = self.csv_path.exists()

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Log a dictionary of metrics for a given step to all destinations (console, wandb, CSV)."""
        parts = [f"Step {step:06d}"]
        if "loss" in metrics:
            parts.append(f"Loss {metrics['loss']:.4f}")

        for k, v in metrics.items():
            if k == "loss":
                continue
            if isinstance(v, float):
                if 0 < v < 1e-3 or v > 1e4:
                    parts.append(f"{k} {v:.2e}")
                else:
                    parts.append(f"{k} {v:.4f}")
            else:
                parts.append(f"{k} {v}")

        logger.info(" | ".join(parts))

        if self.use_wandb:
            import wandb

            wandb.log(metrics, step=step)

        if self.use_csv and self.csv_path is not None:
            row = {"step": step, **metrics}
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                if not self.csv_headers_written:
                    writer.writeheader()
                    self.csv_headers_written = True

                try:
                    writer.writerow(row)
                except ValueError as e:
                    logger.error("Failed to write CSV row due to mismatched keys: %s", e)

    def close(self) -> None:
        """Flush and close all logging streams."""
        if self.use_wandb:
            import wandb

            wandb.finish()
