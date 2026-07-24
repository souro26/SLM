"""
scripts/split_val.py

Carves a validation split out of the existing train.bin without re-running
the data pipeline.

Strategy:
  - Load train.bin as a flat uint16 array (~728M tokens, 1.4GB).
  - Divide into fixed 2048-token chunks (same size as training context windows).
  - Randomly sample 5% of chunks uniformly across the entire file so the
    validation set represents Stage 1/2/3 difficulty proportionally.
  - Write sampled chunks to val.bin; write remaining chunks (preserving
    original order) back to train.bin.

This gives a validation set that exactly mirrors the training objective
(next-token prediction on 2048-token context windows) and is unbiased
with respect to the curriculum stage distribution.

Usage:
    python -m scripts.split_val
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TRAIN_BIN = Path("data/processed/train.bin")
VAL_BIN = Path("data/processed/val.bin")
CHUNK_SIZE = 2048  # tokens — matches context_length in model.yaml
VAL_FRAC = 0.05  # 5% of chunks go to validation
SEED = 42


def main() -> None:
    if not TRAIN_BIN.exists():
        logger.error("train.bin not found at %s", TRAIN_BIN.resolve())
        sys.exit(1)

    if VAL_BIN.exists():
        logger.warning(
            "val.bin already exists at %s — aborting to avoid overwrite.", VAL_BIN.resolve()
        )
        logger.warning("Delete it manually and re-run if you want to re-split.")
        sys.exit(1)

    logger.info("Loading %s ...", TRAIN_BIN.resolve())
    tokens = np.fromfile(TRAIN_BIN, dtype=np.uint16)
    total_tokens = len(tokens)
    logger.info("Loaded %d tokens (%.2f GB)", total_tokens, total_tokens * 2 / 1024**3)

    # Trim any trailing tokens that don't fill a full chunk
    n_chunks = total_tokens // CHUNK_SIZE
    usable_tokens = n_chunks * CHUNK_SIZE
    if usable_tokens < total_tokens:
        logger.info(
            "Trimming %d trailing tokens that don't fill a full chunk.",
            total_tokens - usable_tokens,
        )
    tokens = tokens[:usable_tokens]

    # Sample val chunk indices uniformly across the whole file
    rng = np.random.default_rng(SEED)
    all_indices = np.arange(n_chunks)
    n_val = max(1, int(n_chunks * VAL_FRAC))
    val_indices = set(rng.choice(all_indices, size=n_val, replace=False).tolist())
    train_indices = [i for i in range(n_chunks) if i not in val_indices]

    logger.info(
        "Splitting: %d train chunks / %d val chunks (%.1f%%)",
        len(train_indices),
        n_val,
        n_val / n_chunks * 100,
    )

    # Build arrays
    chunks = tokens.reshape(n_chunks, CHUNK_SIZE)
    val_tokens = chunks[sorted(val_indices)].reshape(-1)
    train_tokens = chunks[train_indices].reshape(-1)

    logger.info(
        "Writing val.bin  — %d tokens (%.2f GB) ...",
        len(val_tokens),
        len(val_tokens) * 2 / 1024**3,
    )
    val_tokens.tofile(VAL_BIN)

    logger.info(
        "Writing train.bin — %d tokens (%.2f GB) ...",
        len(train_tokens),
        len(train_tokens) * 2 / 1024**3,
    )
    train_tokens.tofile(TRAIN_BIN)

    logger.info("Done! val.bin and train.bin are ready.")
    logger.info(
        "Val  : %d chunks × %d tokens = %d tokens",
        n_val,
        CHUNK_SIZE,
        len(val_tokens),
    )
    logger.info(
        "Train: %d chunks × %d tokens = %d tokens",
        len(train_indices),
        CHUNK_SIZE,
        len(train_tokens),
    )


if __name__ == "__main__":
    main()
