"""
train/checkpoint.py

Handles saving and resuming training state. A full training state consists of:
  - Model weights
  - Optimizer states (Muon and AdamW)
  - Training progress (optimizer step)
  - Data stream position (so training resumes exactly where it left off)

Also manages disk space by keeping only the `keep_last_n` most recent checkpoints.

Usage:
    from train.checkpoint import CheckpointManager

    manager = CheckpointManager(cfg.checkpoint_dir, keep_last_n=cfg.keep_last_n)

    # Save
    manager.save(
        step=step,
        model=model,
        muon_opt=muon_optimizer,
        adamw_opt=adamw_optimizer,
        stream_pos=train_stream.position,
        stream_epoch=train_stream.epoch,
    )

    # Load
    state = manager.load(cfg.resume_from, model, muon_opt, adamw_opt)
    start_step = state.get("step", 0)
    train_stream.set_position(state.get("stream_pos", 0))
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import torch
from torch import nn, optim

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving and garbage-collecting training checkpoints."""

    def __init__(self, checkpoint_dir: str | Path, keep_last_n: int = 5) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = max(1, keep_last_n)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        step: int,
        model: nn.Module,
        muon_opt: optim.Optimizer,
        adamw_opt: optim.Optimizer,
        stream_pos: int,
        stream_epoch: int,
    ) -> Path:
        """Save a checkpoint and clean up old ones."""
        step_dir = self.checkpoint_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving checkpoint to %s", step_dir)

        torch.save(
            {
                "model": model.state_dict(),
                "muon": muon_opt.state_dict(),
                "adamw": adamw_opt.state_dict(),
            },
            step_dir / "tensors.pt",
        )

        metadata = {
            "step": step,
            "stream_pos": stream_pos,
            "stream_epoch": stream_epoch,
        }
        with open(step_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self._cleanup()
        return step_dir

    def load(
        self,
        resume_path: str | Path,
        model: nn.Module,
        muon_opt: optim.Optimizer,
        adamw_opt: optim.Optimizer,
    ) -> dict:
        """Load weights and optimizer states in-place."""
        resume_path = Path(resume_path)
        logger.info("Resuming checkpoint from %s", resume_path)

        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

        tensor_path = resume_path / "tensors.pt"
        if not tensor_path.exists():
            if resume_path.is_file() and resume_path.suffix == ".pt":
                tensor_path = resume_path
            else:
                raise FileNotFoundError(f"Missing tensors.pt in {resume_path}")

        state_dict = torch.load(tensor_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict["model"])
        muon_opt.load_state_dict(state_dict["muon"])
        adamw_opt.load_state_dict(state_dict["adamw"])

        meta_path = resume_path / "metadata.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            logger.warning("No metadata.json found in %s, returning empty metadata", resume_path)

        return metadata

    def get_latest_checkpoint(self) -> Path | None:
        """Return the path to the highest-step checkpoint, or None if none exist."""
        if not self.checkpoint_dir.exists():
            return None

        checkpoints = []
        for p in self.checkpoint_dir.iterdir():
            if p.is_dir() and "step_" in p.name:
                try:
                    step_num = int(p.name.split("step_")[1])
                    checkpoints.append((step_num, p))
                except ValueError:
                    pass

        if not checkpoints:
            return None

        checkpoints.sort(key=lambda x: x[0])
        return checkpoints[-1][1]

    def _cleanup(self) -> None:
        """Keep only the `keep_last_n` most recent step_XXXXXX directories."""
        checkpoints = []
        for p in self.checkpoint_dir.iterdir():
            if p.is_dir() and p.name.startswith("step_"):
                try:
                    step_num = int(p.name.split("_")[1])
                    checkpoints.append((step_num, p))
                except ValueError:
                    pass

        checkpoints.sort(key=lambda x: x[0])

        if len(checkpoints) > self.keep_last_n:
            to_delete = checkpoints[: -self.keep_last_n]
            for _, p in to_delete:
                logger.debug("Deleting old checkpoint: %s", p)
                try:
                    shutil.rmtree(p)
                except OSError as e:
                    logger.warning("Failed to delete old checkpoint %s: %s", p, e)
