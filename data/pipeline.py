"""
data/pipeline.py

Orchestrates the full data pipeline end to end:
    1. Stream raw source files from all four sources
    2. Filter for quality
    3. Deduplicate
    4. Assign curriculum stages
    5. Pack into flat binary token array

Curated and stdlib sources are processed FIRST (Option A for dedup quality).
This ensures high-quality files win in near-duplicate collisions — if a NumPy
fork appears later in The Stack stream, it gets discarded, not the real thing.

Output:
    data/processed/tokens.bin       — flat uint16 token array
    data/processed/metadata.json    — stats and stage boundaries

Usage:
    python data/pipeline.py
    python data/pipeline.py --pilot   # small run to verify everything works
"""

import argparse
import time
from collections.abc import Iterator
from pathlib import Path

from data.curriculum import ordered_stream, split_by_stage
from data.deduplicate import Deduplicator
from data.download import (
    stream_curated,
    stream_docs,
    stream_stack,
    stream_stackoverflow,
)
from data.filter import is_quality_file
from data.pack import pack_to_binary, save_metadata
from tokenizer.tokenizer import SLMTokenizer

TOKENIZER_DIR = Path("tokenizer/trained")
OUTPUT_DIR = Path("data/processed")
TOKENS_FILE = OUTPUT_DIR / "tokens.bin"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

FULL_STACK_DOCS = 500_000
FULL_SO_DOCS = 200_000
FULL_DOCS_DOCS = 50_000
FULL_CURATED_DOCS = 30_000

PILOT_STACK_DOCS = 5_000
PILOT_SO_DOCS = 1_000
PILOT_DOCS_DOCS = 500
PILOT_CURATED_DOCS = 300


def stream_all_sources(
    stack_limit: int,
    so_limit: int,
    docs_limit: int,
    curated_limit: int,
) -> Iterator[tuple[str, str]]:
    """Yield (source_text, source_tag) tuples from all four sources."""
    print("--- Streaming curated repos (first, for dedup priority) ---")
    for text in stream_curated(max_docs=curated_limit):
        yield text, "curated"

    print("--- Streaming stdlib + docs ---")
    for text in stream_docs(max_docs=docs_limit):
        yield text, "stdlib"

    print("--- Streaming Stack Overflow ---")
    for text in stream_stackoverflow(max_docs=so_limit):
        yield text, "stackoverflow"

    print("--- Streaming The Stack v2 ---")
    for text in stream_stack(max_docs=stack_limit):
        yield text, "stack"


def build_clean_stream(
    raw_iter: Iterator[tuple[str, str]],
    dedup: Deduplicator,
) -> Iterator[tuple[str, str]]:
    """Apply filtering and deduplication to the raw source stream."""
    for source, tag in raw_iter:
        require_docstring = tag == "stack"

        if not is_quality_file(source, require_docstring=require_docstring):
            continue

        if dedup.is_unique(source):
            yield source, tag


def run_pipeline(pilot: bool = False):
    """Build the complete pipeline."""
    start = time.time()

    if pilot:
        stack_limit = PILOT_STACK_DOCS
        so_limit = PILOT_SO_DOCS
        docs_limit = PILOT_DOCS_DOCS
        curated_limit = PILOT_CURATED_DOCS
        print("=" * 60)
        print("SLM Data Pipeline — PILOT RUN")
    else:
        stack_limit = FULL_STACK_DOCS
        so_limit = FULL_SO_DOCS
        docs_limit = FULL_DOCS_DOCS
        curated_limit = FULL_CURATED_DOCS
        print("=" * 60)
        print("SLM Data Pipeline — FULL RUN")

    total_estimated = stack_limit + so_limit + docs_limit + curated_limit
    print(f"  estimated input files : ~{total_estimated:,}")
    print(f"  output                : {TOKENS_FILE}")
    print("=" * 60)

    print("\nLoading tokenizer...")
    tokenizer = SLMTokenizer(TOKENIZER_DIR)
    print(tokenizer)

    dedup = Deduplicator(threshold=0.85)
    print("\nStreaming, filtering, deduplicating...")
    raw_iter = stream_all_sources(stack_limit, so_limit, docs_limit, curated_limit)
    clean_iter = build_clean_stream(raw_iter, dedup)
    print("\nAssigning curriculum stages...")
    stage1, stage2, stage3 = split_by_stage(clean_iter)
    print(f"\nDedup stats: {dedup.stats}")
    print("\nPacking to binary token array...")
    total_clean = len(stage1) + len(stage2) + len(stage3)
    ordered = ordered_stream(stage1, stage2, stage3)

    stats = pack_to_binary(
        source_iter=ordered,
        tokenizer=tokenizer,
        output_path=TOKENS_FILE,
        estimated_files=total_clean,
    )

    if not stats:
        return

    stage1_tokens = _count_stage_tokens(stage1, tokenizer)
    stage2_tokens = _count_stage_tokens(stage2, tokenizer)

    stage_boundaries = {
        "stage1_end_token": stage1_tokens,
        "stage2_end_token": stage1_tokens + stage2_tokens,
    }

    save_metadata(OUTPUT_DIR, stats, stage_boundaries)

    elapsed = time.time() - start
    print(f"\nPipeline complete in {elapsed / 60:.1f} minutes.")
    print(f"Token array ready at {TOKENS_FILE}")


def _count_stage_tokens(sources: list[str], tokenizer) -> int:
    """Count total tokens across a list of source files."""
    total = 0
    for source in sources:
        total += len(tokenizer.encode(source, add_eof=True))
    return total


def main():
    parser = argparse.ArgumentParser(description="SLM data pipeline")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run with small file limits to verify the pipeline works end to end",
    )
    args = parser.parse_args()
    run_pipeline(pilot=args.pilot)


if __name__ == "__main__":
    main()
