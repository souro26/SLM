"""
train/trainer.py

Main training orchestrator.
Handles the training loop, gradient accumulation, mixed precision, and checkpointing.

Features:
- Graceful Pause: Press Ctrl+C OR create a file named `PAUSE` in the root directory to safely save state and exit.
- Full Resumability: Uses CheckpointManager and PackedTokenStream to resume mid-epoch without skipping tokens.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from train.checkpoint import CheckpointManager
from train.config import TrainConfig
from train.data import PackedTokenStream
from train.logger import TrainLogger
from train.optim import create_optimizers
from train.scheduler import WarmupCosineSchedule

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg: TrainConfig, model: torch.nn.Module) -> None:
        self.cfg = cfg
        self.model = model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Apply gradient checkpointing if requested
        self.model.gradient_checkpointing = getattr(self.cfg, "gradient_checkpointing", False)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        if self.cfg.compile and hasattr(torch, "compile"):
            logger.info("Compiling model...")
            self.model = torch.compile(self.model)

        self.muon_opt, self.adamw_opt = create_optimizers(self.model, self.cfg)
        self.train_stream = PackedTokenStream(
            self.cfg.train_data,
            batch_size=self.cfg.micro_batch_size,
            context_length=self.model.cfg.context_length,
        )

        self.val_stream = None
        if self.cfg.val_data:
            self.val_stream = PackedTokenStream(
                self.cfg.val_data,
                batch_size=self.cfg.micro_batch_size,
                context_length=self.model.cfg.context_length,
            )

        self.total_steps = self.cfg.max_tokens // (
            self.cfg.micro_batch_size * self.cfg.grad_accum_steps * self.model.cfg.context_length
        )
        self.scheduler = WarmupCosineSchedule(
            warmup_steps=self.cfg.lr_schedule.warmup_steps,
            total_steps=self.total_steps,
            min_lr_ratio=self.cfg.lr_schedule.min_lr_ratio,
        )

        self.checkpoint_manager = CheckpointManager(
            self.cfg.checkpoint_dir, keep_last_n=self.cfg.keep_last_n
        )
        self.train_logger = TrainLogger(self.cfg)

        self.start_step = 0
        if self.cfg.resume_from and self.cfg.resume_from.lower() == "latest":
            latest_ckpt = self.checkpoint_manager.get_latest_checkpoint()
            if latest_ckpt:
                self.cfg.resume_from = str(latest_ckpt)
                logger.info("Auto-resolved 'latest' checkpoint to: %s", latest_ckpt)
            else:
                logger.info("resume_from='latest' but no checkpoints found. Starting from scratch.")
                self.cfg.resume_from = None

        if self.cfg.resume_from:
            self._resume()

    def _resume(self) -> None:
        try:
            meta = self.checkpoint_manager.load(
                self.cfg.resume_from, self.model, self.muon_opt, self.adamw_opt
            )
            self.start_step = meta.get("step", 0)

            pos = meta.get("stream_pos", 0)
            self.train_stream.set_position(pos)

            logger.info("Resumed from step %d, stream position %d", self.start_step, pos)
        except Exception as e:
            logger.error("Failed to resume from %s: %s", self.cfg.resume_from, e)
            raise

    @torch.no_grad()
    def _evaluate(self, step: int, num_iters: int = 20) -> None:
        if self.val_stream is None:
            return

        logger.info("Running evaluation at step %d...", step)
        self.model.eval()

        total_loss = 0.0
        for _ in range(num_iters):
            input_ids, target_ids = self.val_stream.next_batch()
            input_ids = input_ids.to(self.device, non_blocking=True)
            target_ids = target_ids.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                logits, _ = self.model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), target_ids.view(-1)
                )
                total_loss += loss.item()

        avg_loss = total_loss / num_iters
        self.train_logger.log_metrics(step, {"val_loss": avg_loss}, tag="eval")

        self.model.train()

    def run(self) -> None:
        logger.info("Starting training loop. Target steps: %d", self.total_steps)
        logger.info(
            "Create a file named 'PAUSE' in the root directory or press Ctrl+C to stop gracefully."
        )

        self.model.train()
        step = self.start_step

        tokens_per_step = (
            self.cfg.micro_batch_size * self.cfg.grad_accum_steps * self.model.cfg.context_length
        )

        pause_file = Path("PAUSE")
        if pause_file.exists():
            pause_file.unlink()

        try:
            while step < self.total_steps:
                if step % self.cfg.eval_every_steps == 0:
                    self._evaluate(step)

                step_start_time = time.time()

                current_muon_lr = self.scheduler.get_lr(step, self.cfg.optimizer.muon.lr)
                current_adamw_lr = self.scheduler.get_lr(step, self.cfg.optimizer.adamw.lr)

                for group in self.muon_opt.param_groups:
                    group["lr"] = current_muon_lr
                for group in self.adamw_opt.param_groups:
                    group["lr"] = current_adamw_lr

                self.muon_opt.zero_grad(set_to_none=True)
                self.adamw_opt.zero_grad(set_to_none=True)

                total_loss = 0.0
                for _ in range(self.cfg.grad_accum_steps):
                    input_ids, target_ids = self.train_stream.next_batch()
                    input_ids = input_ids.to(self.device, non_blocking=True)
                    target_ids = target_ids.to(self.device, non_blocking=True)
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        logits, _ = self.model(input_ids)
                        loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)), target_ids.view(-1)
                        )
                        loss = loss / self.cfg.grad_accum_steps

                    loss.backward()
                    total_loss += loss.item()

                if math.isnan(total_loss) or math.isinf(total_loss):
                    logger.error(
                        "NaN or Inf loss detected! Saving emergency checkpoint and halting."
                    )
                    self._save_checkpoint(step, prefix="emergency_NaN")
                    sys.exit(1)

                # TrainConfig guarantees grad_clip > 0, so this always runs
                grad_norm = clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

                self.muon_opt.step()
                self.adamw_opt.step()

                step += 1

                if step % self.cfg.log_every_steps == 0 or step == self.total_steps:
                    dt = time.time() - step_start_time
                    tokens_per_sec = tokens_per_step / dt if dt > 0 else 0.0

                    self.train_logger.log_metrics(
                        step,
                        {
                            "loss": total_loss,
                            "lr_muon": current_muon_lr,
                            "lr_adamw": current_adamw_lr,
                            "grad_norm": (
                                grad_norm.item()
                                if isinstance(grad_norm, torch.Tensor)
                                else grad_norm
                            ),
                            "dt_sec": dt,
                            "tokens_per_sec": tokens_per_sec,
                        },
                    )

                if step % self.cfg.save_every_steps == 0:
                    self._save_checkpoint(step)

                if pause_file.exists():
                    logger.info("PAUSE file detected. Safely stopping training.")
                    self._save_checkpoint(step, prefix="paused")
                    pause_file.unlink()
                    break

        except KeyboardInterrupt:
            logger.info(
                "\nKeyboard interrupt received! Saving emergency checkpoint before exiting..."
            )
            self._save_checkpoint(step, prefix="interrupted")
        except torch.cuda.OutOfMemoryError as e:
            logger.error("CUDA Out of Memory at step %d! Printing memory stats:", step)
            logger.error(torch.cuda.memory_summary(device=self.device))
            self._save_checkpoint(step, prefix="OOM")
            raise e
        except Exception as e:
            logger.error("Unexpected error at step %d: %s", step, e)
            self._save_checkpoint(step, prefix="crash")
            raise e
        finally:
            self.train_logger.close()
            logger.info("Training loop ended at step %d", step)

    def _save_checkpoint(self, step: int, prefix: str | None = None) -> None:
        """Helper to save state with the manager, allowing custom prefixes for emergency saves."""
        current_muon_lr = self.scheduler.get_lr(step, self.cfg.optimizer.muon.lr)
        current_adamw_lr = self.scheduler.get_lr(step, self.cfg.optimizer.adamw.lr)

        saved_path = self.checkpoint_manager.save(
            step=step,
            model=self.model,
            muon_opt=self.muon_opt,
            adamw_opt=self.adamw_opt,
            stream_pos=self.train_stream.position,
            stream_epoch=self.train_stream.epoch,
            lr_muon=current_muon_lr,
            lr_adamw=current_adamw_lr,
        )

        if prefix:
            # We explicitly avoid the word "step_" so CheckpointManager._cleanup() ignores this directory
            # e.g. crash_000500 instead of crash_step_000500
            safe_name = f"{prefix}_{step:06d}"
            new_path = saved_path.parent / safe_name
            if new_path.exists():
                import shutil

                shutil.rmtree(new_path)
            saved_path.rename(new_path)
            logger.info("Saved special checkpoint to %s", new_path)
