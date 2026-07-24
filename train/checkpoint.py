"""
train/checkpoint.py

Handles saving and resuming training state. A full training state consists of:
  - Model weights
  - Optimizer states (Muon and AdamW)
  - RNG state (CPU and CUDA)
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
        lr_muon=current_muon_lr,
        lr_adamw=current_adamw_lr,
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


def _parse_step(path: Path) -> int | None:
    """Helper to safely extract the step number from a checkpoint directory name."""
    if not path.is_dir() or "step_" not in path.name:
        return None
    try:
        return int(path.name.split("step_")[1])
    except ValueError:
        return None


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
        lr_muon: float = 0.0,
        lr_adamw: float = 0.0,
    ) -> Path:
        """Save a checkpoint atomically and clean up old ones."""
        final_dir = self.checkpoint_dir / f"step_{step:06d}"
        tmp_dir = self.checkpoint_dir / f"step_{step:06d}.tmp"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        tmp_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving checkpoint to %s", final_dir)

        # Get RNG states for reproducibility
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []

        torch.save(
            {
                "model": model.state_dict(),
                "muon": muon_opt.state_dict(),
                "adamw": adamw_opt.state_dict(),
                "rng_state": rng_state,
                "cuda_rng_state": cuda_rng_state,
            },
            tmp_dir / "tensors.pt",
        )

        metadata = {
            "step": step,
            "stream_pos": stream_pos,
            "stream_epoch": stream_epoch,
            "lr_muon": lr_muon,
            "lr_adamw": lr_adamw,
        }
        with open(tmp_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Atomic rename ensures a partially written checkpoint is never picked up
        if final_dir.exists():
            shutil.rmtree(final_dir)
        tmp_dir.rename(final_dir)

        self._cleanup()
        return final_dir

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

        # weights_only=False is intentional here: optimizer state dicts contain Python objects (e.g. step counts)
        # that weights_only=True would reject.
        state_dict = torch.load(tensor_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict["model"])
        muon_opt.load_state_dict(state_dict["muon"])
        adamw_opt.load_state_dict(state_dict["adamw"])

        if "rng_state" in state_dict:
            torch.set_rng_state(state_dict["rng_state"])
        if (
            "cuda_rng_state" in state_dict
            and torch.cuda.is_available()
            and state_dict["cuda_rng_state"]
        ):
            try:
                torch.cuda.set_rng_state_all(state_dict["cuda_rng_state"])
            except Exception as e:
                logger.warning("Could not restore CUDA RNG state: %s", e)

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
            step_num = _parse_step(p)
            if step_num is not None and not p.name.endswith(".tmp"):
                checkpoints.append((step_num, p))

        if not checkpoints:
            return None

        checkpoints.sort(key=lambda x: x[0])
        return checkpoints[-1][1]

    def _cleanup(self) -> None:
        """Keep only the `keep_last_n` most recent step_XXXXXX directories."""
        checkpoints = []
        for p in self.checkpoint_dir.iterdir():
            step_num = _parse_step(p)
            if step_num is not None and not p.name.endswith(".tmp"):
                checkpoints.append((step_num, p))

        checkpoints.sort(key=lambda x: x[0])

        if len(checkpoints) > self.keep_last_n:
            to_delete = checkpoints[: -self.keep_last_n]
            for _, p in to_delete:
                logger.debug("Deleting old checkpoint: %s", p)
                try:
                    shutil.rmtree(p)
                except OSError as e:
                    logger.warning("Failed to delete old checkpoint %s: %s", p, e)
