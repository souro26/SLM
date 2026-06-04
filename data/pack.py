"""
data/pack.py

Tokenizes all source files and packs them into a flat binary token array.
This is the final step of the data pipeline — the output is what training reads.

What this file produces:
    data/processed/tokens.bin   — flat array of uint16 token IDs
    data/processed/metadata.json — token count, stage boundaries, file count

Why a flat binary array:
    Training reads 2048-token chunks sequentially. A flat binary array lets
    you seek to any position instantly and read chunks at full disk speed.
    No padding, no wasted tokens, no per-sample overhead.

    uint16 can represent values 0–65535. Our vocab is 32k so uint16 is enough
    and uses 2 bytes per token. 10B tokens = ~20GB on disk.

Size confirmation:
    Before writing, this script prints the estimated file size and asks you
    to confirm before touching your disk.

Usage:
    from data.pack import pack_to_binary

    pack_to_binary(source_iter, tokenizer, output_path)
"""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def estimate_size_gb(num_tokens: int) -> float:
    """uint16 = 2 bytes per token."""
    return (num_tokens * 2) / (1024**3)


def estimate_token_count(num_files: int, avg_tokens_per_file: int = 512) -> int:
    """Rough estimate of total tokens before we've processed anything."""
    return num_files * avg_tokens_per_file


def pack_to_binary(
    source_iter: Iterator[str],
    tokenizer,
    output_path: Path,
    estimated_files: int = 0,
    chunk_size: int = 1000,
) -> dict:
    """Tokenize all source files and write token IDs to a flat binary file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if estimated_files > 0:
        estimated_tokens = estimate_token_count(estimated_files)
        estimated_gb = estimate_size_gb(estimated_tokens)
        print(f"\nEstimated output size: {estimated_gb:.1f} GB")
        print(f"Output path: {output_path.resolve()}")
        confirm = input("Write to disk? (y/n): ").strip().lower()
        if confirm != "y":
            print("Aborted. Nothing written.")
            return {}

    total_tokens = 0
    total_files = 0
    buffer = []

    print(f"\nPacking tokens to {output_path}...")

    with open(output_path, "wb") as f:
        for source in source_iter:
            buffer.append(source)
            total_files += 1

            if len(buffer) >= chunk_size:
                tokens_written = _flush_buffer(buffer, tokenizer, f)
                total_tokens += tokens_written
                buffer = []

                if total_files % 10_000 == 0:
                    size_gb = estimate_size_gb(total_tokens)
                    print(
                        f"  packed {total_files:,} files, "
                        f"{total_tokens:,} tokens, "
                        f"{size_gb:.2f} GB written"
                    )

        if buffer:
            total_tokens += _flush_buffer(buffer, tokenizer, f)

    size_gb = estimate_size_gb(total_tokens)
    print("\nPack complete.")
    print(f"  files   : {total_files:,}")
    print(f"  tokens  : {total_tokens:,}")
    print(f"  size    : {size_gb:.2f} GB")
    print(f"  path    : {output_path.resolve()}")

    return {
        "total_tokens": total_tokens,
        "total_files": total_files,
        "output_path": str(output_path.resolve()),
        "size_gb": size_gb,
    }


def _flush_buffer(buffer: list[str], tokenizer, file_handle) -> int:
    """Encode a batch of source strings and write their token IDs to disk."""
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
    stage_boundaries: dict = None,
):
    """Save metadata alongside the binary token file."""
    output_dir = Path(output_dir)
    meta = {**stats}
    if stage_boundaries:
        meta["stage_boundaries"] = stage_boundaries

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Metadata saved to {meta_path}")


def load_token_array(tokens_path: Path) -> np.ndarray:
    """Load the packed token array from disk into a numpy array."""
    tokens_path = Path(tokens_path)
    if not tokens_path.exists():
        raise FileNotFoundError(
            f"Token array not found at {tokens_path}. " "Run data/pipeline.py first."
        )

    arr = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    return arr


def get_token_count(tokens_path: Path) -> int:
    """Return the number of tokens in the packed binary file."""
    tokens_path = Path(tokens_path)
    size_bytes = tokens_path.stat().st_size
    return size_bytes // 2
