"""
data/pack.py

Tokenizes all source files and packs them into a flat binary token array.
This is the final step of the data pipeline — the output is what training reads.

What this file produces:
    data/processed/train.bin     — flat array of uint16 token IDs (training)
    data/processed/val.bin       — flat array of uint16 token IDs (validation)
    data/processed/metadata.json — token count, stage boundaries, file count

Why a flat binary array:
    Training reads 2048-token chunks sequentially. A flat binary array lets
    you seek to any position instantly and read chunks at full disk speed.
    No padding, no wasted tokens, no per-sample overhead.

    uint16 can represent values 0-65535. Our vocab is 32k so uint16 is enough
    and uses 2 bytes per token. 10B tokens = ~20GB on disk.

Note on confirmation:
    pack_to_binary does NOT prompt for confirmation interactively. Pass
    dry_run=True to estimate size without writing. The pipeline script
    handles user confirmation before calling this function.

Usage:
    from data.pack import pack_to_binary, load_token_array

    stats = pack_to_binary(source_iter, tokenizer, output_path)
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_LOG_EVERY_FILES = 10_000


def estimate_size_gb(num_tokens: int) -> float:
    """uint16 = 2 bytes per token."""
    return (num_tokens * 2) / (1024**3)


def estimate_token_count(num_files: int, avg_tokens_per_file: int = 512) -> int:
    """Estimate total tokens before processing anything."""
    return num_files * avg_tokens_per_file


def pack_to_binary(
    source_iter: Iterator[str],
    tokenizer,
    output_path: Path,
    estimated_files: int = 0,
    chunk_size: int = 1000,
    dry_run: bool = False,
) -> dict:
    """Tokenize source files to a binary file."""
    output_path = Path(output_path)

    if estimated_files > 0:
        estimated_tokens = estimate_token_count(estimated_files)
        estimated_gb = estimate_size_gb(estimated_tokens)
        logger.info(
            "pack estimate: ~%d files, ~%d tokens, ~%.1f GB at %s",
            estimated_files,
            estimated_tokens,
            estimated_gb,
            output_path.resolve(),
        )

    if dry_run:
        logger.info("dry_run=True — nothing written")
        return {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("pack_to_binary: writing to %s", output_path.resolve())

    total_tokens = 0
    total_files = 0
    buffer: list[str] = []

    with open(output_path, "wb") as f:
        for source in source_iter:
            buffer.append(source)
            total_files += 1

            if len(buffer) >= chunk_size:
                tokens_written = _flush_buffer(buffer, tokenizer, f)
                total_tokens += tokens_written
                buffer = []

            if total_files % _LOG_EVERY_FILES == 0:
                size_gb = estimate_size_gb(total_tokens)
                logger.info(
                    "pack: %d files, %d tokens, %.2f GB written",
                    total_files,
                    total_tokens,
                    size_gb,
                )

        if buffer:
            total_tokens += _flush_buffer(buffer, tokenizer, f)

    size_gb = estimate_size_gb(total_tokens)
    logger.info(
        "pack done: %d files, %d tokens, %.2f GB at %s",
        total_files,
        total_tokens,
        size_gb,
        output_path.resolve(),
    )

    return {
        "total_tokens": total_tokens,
        "total_files": total_files,
        "output_path": str(output_path.resolve()),
        "size_gb": size_gb,
    }


def _flush_buffer(buffer: list[str], tokenizer, file_handle) -> int:
    """Encode and write a batch to disk."""
    ids_list = tokenizer.encode_batch(buffer, add_eof=True)

    tokens_written = 0
    for ids in ids_list:
        arr = np.array(ids, dtype=np.uint16)
        file_handle.write(arr.tobytes())
        tokens_written += len(ids)

    return tokens_written


def save_metadata(
    output_dir: Path,
    stats: dict,
    stage_boundaries: dict | None = None,
) -> None:
    """Save metadata alongside the binary file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {**stats}
    if stage_boundaries:
        meta["stage_boundaries"] = stage_boundaries

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("metadata saved to %s", meta_path)


def load_token_array(tokens_path: Path) -> np.ndarray:
    """Load packed token array as memory-mapped array."""
    tokens_path = Path(tokens_path)
    if not tokens_path.exists():
        raise FileNotFoundError(
            f"Token array not found at {tokens_path}. Run data/pipeline.py first."
        )

    arr = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    logger.info(
        "loaded token array: %d tokens (%.2f GB) from %s",
        len(arr),
        estimate_size_gb(len(arr)),
        tokens_path,
    )
    return arr


def get_token_count(tokens_path: Path) -> int:
    """Return token count without loading the file."""
    tokens_path = Path(tokens_path)
    size_bytes = tokens_path.stat().st_size
    return size_bytes // 2
