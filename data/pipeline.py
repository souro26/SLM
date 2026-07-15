"""
data/pipeline.py

Orchestrates the full data pipeline end to end:
    1. Stream raw source files from all four sources
    2. Filter for quality
    3. Deduplicate
    4. Assign curriculum stages
    5. Pack into flat binary token array

Output:
    data/processed/train.bin              — flat uint16 token array
    data/processed/metadata.json          — stats and stage boundaries
    data/processed/clean_checkpoint.jsonl — incremental checkpoint of
                                             clean (source, tag) pairs

Usage:
    python -m data.pipeline                 # full run
    python -m data.pipeline --pilot         # small run to verify end to end
    python -m data.pipeline --pilot --yes   # skip confirmation prompt
    python -m data.pipeline --resume        # resume from last checkpoint
                                             # instead of re-streaming
"""

import argparse
import json
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
CHECKPOINT_FILE = OUTPUT_DIR / "clean_checkpoint.jsonl"

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
    checkpoint_path: Path | None = None,
) -> Iterator[tuple[str, str]]:
    """Apply filtering and deduplication to source stream.

    v4: every per-file operation is wrapped in try/except — any
    unexpected error (RecursionError, malformed input, anything we
    haven't anticipated) logs a warning and skips that one file rather
    than propagating and killing the entire run. Every file that passes
    is also immediately appended to checkpoint_path (if given) so
    progress survives a crash anywhere downstream.
    """
    if checkpoint_path is None:
        yield from _clean_stream_body(raw_iter, dedup, checkpoint_fh=None)
        return

    with open(checkpoint_path, "a", encoding="utf-8") as checkpoint_fh:
        yield from _clean_stream_body(raw_iter, dedup, checkpoint_fh=checkpoint_fh)


def _clean_stream_body(
    raw_iter: Iterator[tuple[str, str]],
    dedup: Deduplicator,
    checkpoint_fh,
) -> Iterator[tuple[str, str]]:
    """Shared filtering/dedup/checkpoint logic, called with or without a checkpoint file open."""
    skipped_errors = 0

    for source, tag in raw_iter:
        try:
            require_docstring = tag == "stack"

            if not is_quality_file(source, require_docstring=require_docstring):
                continue

            if not dedup.is_unique(source):
                continue

            if checkpoint_fh is not None:
                checkpoint_fh.write(json.dumps({"source": source, "tag": tag}) + "\n")
                checkpoint_fh.flush()

            yield source, tag

        except Exception as err:  # noqa: BLE001 - one bad file must never kill the run
            skipped_errors += 1
            logger.warning(
                "build_clean_stream: skipped one file after unexpected error "
                "(tag=%s, %d skipped so far): %s",
                tag,
                skipped_errors,
                err,
            )
            continue


def load_checkpoint(checkpoint_path: Path) -> Iterator[tuple[str, str]]:
    """Load clean (source, tag) pairs from a checkpoint file for --resume."""
    loaded = 0
    skipped_bad_lines = 0
    with open(checkpoint_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                loaded += 1
                yield obj["source"], obj["tag"]
            except (json.JSONDecodeError, KeyError):
                skipped_bad_lines += 1
                continue
    logger.info(
        "load_checkpoint: loaded %d clean files from %s (%d bad lines skipped)",
        loaded,
        checkpoint_path,
        skipped_bad_lines,
    )


def run_pipeline(pilot: bool = False, yes: bool = False, resume: bool = False) -> None:
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
    logger.info("SLM Data Pipeline — %s%s", label, " (RESUME)" if resume else "")
    logger.info("  estimated input files : ~%d", total_estimated)
    logger.info("  output                : %s", TRAIN_FILE.resolve())
    logger.info("  checkpoint            : %s", CHECKPOINT_FILE.resolve())
    logger.info("=" * 60)

    if resume and not CHECKPOINT_FILE.exists():
        logger.error("--resume given but no checkpoint file found at %s", CHECKPOINT_FILE.resolve())
        return

    if not pilot and not yes and not resume:
        print(f"\nFull run will write up to ~20GB to {TRAIN_FILE.resolve()}")
        confirm = input("Proceed? (y/n): ").strip().lower()
        if confirm != "y":
            logger.info("aborted by user")
            return

    logger.info("loading tokenizer from %s", TOKENIZER_DIR)
    tokenizer = SLMTokenizer(TOKENIZER_DIR)
    logger.info("tokenizer loaded: %s", tokenizer)

    if resume:
        logger.info("resuming from checkpoint, skipping source streaming entirely")
        clean_iter = load_checkpoint(CHECKPOINT_FILE)
    else:
        # Fresh run: start checkpoint file clean (don't append to stale
        # data from an unrelated previous run).
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_FILE.write_text("", encoding="utf-8")

        dedup = Deduplicator(threshold=0.85)

        logger.info("streaming, filtering, deduplicating...")
        raw_iter = stream_all_sources(stack_limit, so_limit, docs_limit, curated_limit)
        clean_iter = build_clean_stream(raw_iter, dedup, checkpoint_path=CHECKPOINT_FILE)

    logger.info("assigning curriculum stages...")
    stage1, stage2, stage3 = split_by_stage(clean_iter)

    total_clean = len(stage1) + len(stage2) + len(stage3)
    logger.info(
        "clean files: %d total (stage1=%d, stage2=%d, stage3=%d)",
        total_clean,
        len(stage1),
        len(stage2),
        len(stage3),
    )

    if total_clean == 0:
        logger.warning("no clean files collected — nothing to pack")
        return

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

    save_metadata(OUTPUT_DIR, {**stats, "pilot": pilot, "resumed": resume}, stage_boundaries)

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
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from data/processed/clean_checkpoint.jsonl instead of "
            "re-streaming all sources. Use after a crash to pack whatever "
            "was already collected instead of starting over."
        ),
    )
    args = parser.parse_args()
    run_pipeline(pilot=args.pilot, yes=args.yes, resume=args.resume)


if __name__ == "__main__":
    main()
