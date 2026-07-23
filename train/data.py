"""
train/data.py

Streams fixed-length token chunks from a packed binary token array
(data/processed/train.bin or val.bin) for language-model training.

Deliberately sequential, not randomly sampled. The corpus was packed with
curriculum ordering baked in (stage1 simple -> stage2 complex -> stage3
upsampled cooldown, per data/curriculum.py and data/pack.py) — randomly
sampling offsets would destroy that ordering. Checkpoints track a single
token-index cursor into the flat array (see position/set_position below),
which only makes sense for sequential reading; this is why the design
uses a hand-rolled stateful stream instead of torch.utils.data.Dataset +
DataLoader, whose default shuffling assumption is the wrong model here.

Batch construction: each row in a batch is a context_length-token window;
row i starts context_length tokens after row i-1 (rows overlap by exactly
one token at the x/y boundary — row i's target for its last position is
row i+1's first input token — this is normal, not a bug). The cursor
advances by batch_size * context_length tokens per call to next_batch().

Wraparound: if the corpus is exhausted mid-batch, the cursor wraps to 0
and an epoch counter increments (logged) — a full pass through 728M
tokens is expected to happen multiple times over a multi-billion-token
training run.

Usage:
    from train.data import PackedTokenStream

    train_stream = PackedTokenStream(
        "data/processed/train.bin", context_length=2048, batch_size=4,
    )
    x, y = train_stream.next_batch()  # [4, 2048] each, dtype long

    # checkpoint resume:
    saved_position = train_stream.position
    ...
    train_stream.set_position(saved_position)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_PACKED_DTYPE = np.uint16


class PackedTokenStream:
    """Sequential, resumable reader over a flat packed uint16 token array."""

    def __init__(
        self,
        bin_path: str | Path,
        context_length: int,
        batch_size: int,
        start_position: int = 0,
    ) -> None:
        bin_path = Path(bin_path)
        if not bin_path.exists():
            raise FileNotFoundError(
                f"packed token file not found: '{bin_path}' — has data/pack.py "
                "been run, and does this split (train/val) actually exist yet?"
            )

        self.bin_path = bin_path
        self.context_length = context_length
        self.batch_size = batch_size

        self._data = np.memmap(bin_path, dtype=_PACKED_DTYPE, mode="r")
        self.n_tokens = self._data.shape[0]

        min_required = context_length + 1
        if self.n_tokens < min_required:
            raise ValueError(
                f"'{bin_path}' has only {self.n_tokens} tokens, but a single "
                f"context_length={context_length} chunk needs at least "
                f"{min_required} (context_length + 1 for the shifted target)"
            )

        self._pos = 0
        self._epoch = 0
        self.set_position(start_position)

    @property
    def position(self) -> int:
        """Current token-index cursor — save this in checkpoints to resume."""
        return self._pos

    @property
    def epoch(self) -> int:
        """Number of full wraparounds through the corpus so far."""
        return self._epoch

    def set_position(self, pos: int) -> None:
        """Jump the cursor to an absolute token index (e.g. on checkpoint resume)."""
        if not (0 <= pos < self.n_tokens):
            raise ValueError(
                f"position {pos} out of range [0, {self.n_tokens}) for '{self.bin_path}'"
            )
        self._pos = pos

    def _next_row(self) -> tuple[np.ndarray, np.ndarray]:
        """Fetch one context_length row, wrapping the cursor if needed."""
        needed = self.context_length + 1
        if self._pos + needed > self.n_tokens:
            self._epoch += 1
            logger.info(
                "%s: reached end of corpus (%d tokens), wrapping to start — epoch %d",
                self.bin_path,
                self.n_tokens,
                self._epoch,
            )
            self._pos = 0

        start = self._pos
        end = start + needed
        chunk = self._data[start:end]
        x = chunk[:-1]
        y = chunk[1:]
        self._pos += self.context_length
        return x, y

    def next_batch(self, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next (x, y) batch, each [batch_size, context_length], dtype long."""
        xs, ys = [], []
        for _ in range(self.batch_size):
            x, y = self._next_row()
            xs.append(x)
            ys.append(y)

        x_batch = torch.from_numpy(np.stack(xs).astype(np.int64))
        y_batch = torch.from_numpy(np.stack(ys).astype(np.int64))

        if device != "cpu":
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

        return x_batch, y_batch
