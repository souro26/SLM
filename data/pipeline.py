"""
data/pipeline.py  (v3)

Orchestrates the full data pipeline end to end:
    1. Stream raw source files from all four sources
    2. Filter for quality
    3. Deduplicate
    4. Assign curriculum stages
    5. Pack into flat binary token array

v3 change: curated (NumPy, PyTorch, pandas, FastAPI, requests) and
stdlib (cpython, typeshed) sources are now fetched DIRECTLY from GitHub
via fetch_curated_repos() / fetch_stdlib_repos() instead of being hunted
for inside The Stack's streamed shards. The Stack is only scanned once,
for the general 70% sample, and exits as soon as that quota is hit —
no more waiting on rare-repo hits buried in 206 shards.

Output:
    data/processed/train.bin        — flat uint16 token array
    data/processed/metadata.json    — stats and stage boundaries

Usage:
    python -m data.pipeline                 # full run
    python -m data.pipeline --pilot         # small run to verify end to end
    python -m data.pipeline --pilot --yes   # skip confirmation prompt
"""

import argparse
import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from data.curriculum import STAGE3_UPSAMPLE, ordered_stream, split_by_stage
from data.deduplicate import Deduplicator
from data.download import (
    fetch_curated_repos,
    fetch_stdlib_repos,
    stream_stack_all,
    stream_stackoverflow,
)
from data.filter import is_quality_file
from data.pack import pack_to_binary, save_metadata
from tokenizer.tokenizer import SLMTokenizer

TOKENIZER_DIR = Path("tokenizer/trained")
OUTPUT_DIR = Path("data/processed")
TRAIN_FILE = OUTPUT_DIR / "train.bin"

FULL_STACK_DOCS = 500_000
FULL_SO_DOCS = 200_000
FULL_DOCS_DOCS = 50_000
FULL_CURATED_DOCS = 30_000

PILOT_STACK_DOCS = 5_000
PILOT_SO_DOCS = 1_000
PILOT_DOCS_DOCS = 500
PILOT_CURATED_DOCS = 300


def setup_logging(output_dir: Path) -> None:
    """Configure logging for the entire pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(file_handler)

    logging.info("logging initialised. log file: %s", log_path.resolve())


logger = logging.getLogger(__name__)


def stream_all_sources(
    stack_limit: int,
    so_limit: int,
    docs_limit: int,
    curated_limit: int,
) -> Iterator[tuple[str, str]]:
    """Yield (text, tag) tuples from all four sources.

    Curated and stdlib are fetched directly from GitHub first — this is
    fast (a handful of tarball downloads, no scanning). The Stack general
    sample and Stack Overflow are streamed as before.
    """
    logger.info("--- fetching curated repos directly from GitHub ---")
    for text in fetch_curated_repos(limit=curated_limit):
        yield text, "curated"

    logger.info("--- fetching stdlib repos directly from GitHub ---")
    for text in fetch_stdlib_repos(limit=docs_limit):
        yield text, "stdlib"

    logger.info("--- streaming The Stack v1 (general sample only) ---")
    for text in stream_stack_all(stack_limit=stack_limit):
        yield text, "stack"

    logger.info("--- streaming Stack Overflow ---")
    for text in stream_stackoverflow(max_docs=so_limit):
        yield text, "stackoverflow"


def build_clean_stream(
    raw_iter: Iterator[tuple[str, str]],
    dedup: Deduplicator,
) -> Iterator[tuple[str, str]]:
    """Apply filtering and deduplication to source stream."""
    for source, tag in raw_iter:
        require_docstring = tag == "stack"

        if not is_quality_file(source, require_docstring=require_docstring):
            continue

        if dedup.is_unique(source):
            yield source, tag


def run_pipeline(pilot: bool = False, yes: bool = False) -> None:
    """Run the complete data pipeline."""
    start = time.time()

    setup_logging(OUTPUT_DIR)

    if pilot:
        stack_limit = PILOT_STACK_DOCS
        so_limit = PILOT_SO_DOCS
        docs_limit = PILOT_DOCS_DOCS
        curated_limit = PILOT_CURATED_DOCS
        label = "PILOT RUN"
    else:
        stack_limit = FULL_STACK_DOCS
        so_limit = FULL_SO_DOCS
        docs_limit = FULL_DOCS_DOCS
        curated_limit = FULL_CURATED_DOCS
        label = "FULL RUN"

    total_estimated = stack_limit + so_limit + docs_limit + curated_limit

    logger.info("=" * 60)
    logger.info("SLM Data Pipeline — %s", label)
    logger.info("  estimated input files : ~%d", total_estimated)
    logger.info("  output                : %s", TRAIN_FILE.resolve())
    logger.info("=" * 60)

    if not pilot and not yes:
        print(f"\nFull run will write up to ~20GB to {TRAIN_FILE.resolve()}")
        confirm = input("Proceed? (y/n): ").strip().lower()
        if confirm != "y":
            logger.info("aborted by user")
            return

    logger.info("loading tokenizer from %s", TOKENIZER_DIR)
    tokenizer = SLMTokenizer(TOKENIZER_DIR)
    logger.info("tokenizer loaded: %s", tokenizer)

    dedup = Deduplicator(threshold=0.85)

    logger.info("streaming, filtering, deduplicating...")
    raw_iter = stream_all_sources(stack_limit, so_limit, docs_limit, curated_limit)
    clean_iter = build_clean_stream(raw_iter, dedup)

    logger.info("assigning curriculum stages...")
    stage1, stage2, stage3 = split_by_stage(clean_iter)

    logger.info("dedup stats: %s", dedup.stats)

    total_clean = len(stage1) + len(stage2) + len(stage3)
    logger.info(
        "clean files: %d total (stage1=%d, stage2=%d, stage3=%d)",
        total_clean,
        len(stage1),
        len(stage2),
        len(stage3),
    )

    logger.info("counting stage token boundaries...")
    stage1_tokens = _count_tokens(stage1, tokenizer)
    stage2_tokens = _count_tokens(stage2, tokenizer)
    stage3_tokens = _count_tokens(stage3, tokenizer)

    stage_boundaries = {
        "stage1_end_token": stage1_tokens,
        "stage2_end_token": stage1_tokens + stage2_tokens,
        "stage3_end_token": stage1_tokens + stage2_tokens + stage3_tokens * STAGE3_UPSAMPLE,
        "stage3_upsample": STAGE3_UPSAMPLE,
    }
    logger.info("stage boundaries: %s", stage_boundaries)

    logger.info("packing to binary token array...")
    ordered = ordered_stream(stage1, stage2, stage3)

    stats = pack_to_binary(
        source_iter=ordered,
        tokenizer=tokenizer,
        output_path=TRAIN_FILE,
        estimated_files=total_clean,
    )

    if not stats:
        logger.warning("pack_to_binary returned empty — something went wrong")
        return

    save_metadata(OUTPUT_DIR, {**stats, "pilot": pilot}, stage_boundaries)

    elapsed = time.time() - start
    logger.info("pipeline complete in %.1f minutes", elapsed / 60)
    logger.info("token array ready at %s", TRAIN_FILE.resolve())


def _count_tokens(sources: list[str], tokenizer) -> int:
    """Count total tokens across source strings."""
    total = 0
    for source in sources:
        total += len(tokenizer.encode(source, add_eof=True))
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="SLM data pipeline")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run with small file limits to verify the pipeline end to end",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt (for unattended runs)",
    )
    args = parser.parse_args()
    run_pipeline(pilot=args.pilot, yes=args.yes)


if __name__ == "__main__":
    main()
