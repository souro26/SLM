"""
data/download.py

Streams raw text from all four data sources. Writes nothing to disk.
Each function is a generator — yields one document at a time.

Sources:
    - The Stack v1 (Python)                          70% of final mix
    - Stack Overflow Python Q&A                      15%
    - Python docs + stdlib source (cpython/typeshed) 10%
    - Curated open source (NumPy, PyTorch, etc.)      5%

Key design decision:
    The Stack v1 is scanned ONCE via stream_stack_all(), which yields
    (content, tag) tuples routing each file to stack/stdlib/curated
    in a single pass. Previously three separate functions each did a
    full scan — 3x the downloads, 3x the network fragility.

Confirmed field names (verified against live datasets):
    The Stack v1:   content, max_stars_repo_name, max_stars_count,
                    max_line_length, avg_line_length
    SO posts:       Id, PostTypeId, AcceptedAnswerId, ParentId,
                    Score, Body, Tags

Usage:
    from data.download import stream_stack_all, stream_stackoverflow

    for text, tag in stream_stack_all(stack_limit=5000, docs_limit=500, curated_limit=300):
        print(tag, text[:80])
"""

import logging
from collections.abc import Iterator

logger = logging.getLogger(__name__)

_LOG_EVERY = 10_000

_STDLIB_REPOS = {"python/cpython", "python/typeshed"}

_CURATED_REPOS = {
    "numpy/numpy",
    "pytorch/pytorch",
    "pandas-dev/pandas",
    "tiangolo/fastapi",
    "psf/requests",
}


def stream_stack_all(
    stack_limit: int = 500_000,
    docs_limit: int = 50_000,
    curated_limit: int = 30_000,
) -> Iterator[tuple[str, str]]:
    """Single-pass stream over The Stack v1."""
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("pip install datasets") from err

    logger.info(
        "stream_stack_all: starting single pass "
        "(stack_limit=%d, docs_limit=%d, curated_limit=%d)",
        stack_limit,
        docs_limit,
        curated_limit,
    )

    ds = load_dataset(
        "bigcode/the-stack",
        data_dir="data/python",
        split="train",
        streaming=True,
    )

    stack_count = 0
    docs_count = 0
    curated_count = 0
    scanned = 0
    skipped = 0

    for sample in ds:
        scanned += 1

        if (
            stack_count >= stack_limit
            and docs_count >= docs_limit
            and curated_count >= curated_limit
        ):
            break

        content = sample.get("content") or ""
        if not content:
            skipped += 1
            continue

        repo = sample.get("max_stars_repo_name") or ""

        if repo in _STDLIB_REPOS:
            if docs_count >= docs_limit:
                continue
            docs_count += 1
            yield content, "stdlib"

        elif repo in _CURATED_REPOS:
            if curated_count >= curated_limit:
                continue
            curated_count += 1
            yield content, "curated"

        else:
            if stack_count >= stack_limit:
                continue
            stack_count += 1
            yield content, "stack"

        if scanned % _LOG_EVERY == 0:
            logger.info(
                "stream_stack_all: %d scanned — stack=%d, stdlib=%d, curated=%d, skipped=%d",
                scanned,
                stack_count,
                docs_count,
                curated_count,
                skipped,
            )

    logger.info(
        "stream_stack_all done: %d scanned — stack=%d, stdlib=%d, curated=%d, skipped=%d",
        scanned,
        stack_count,
        docs_count,
        curated_count,
        skipped,
    )


def stream_stackoverflow(max_docs: int = 200_000) -> Iterator[str]:
    """Stream Python Q&A pairs from Stack Overflow."""
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("pip install datasets") from err

    max_collect = max_docs * 5

    logger.info(
        "stream_stackoverflow: starting (max_docs=%d, max_collect=%d)",
        max_docs,
        max_collect,
    )

    accepted: dict[int, str] = {}
    scanned_pass1 = 0

    ds = load_dataset(
        "mikex86/stackoverflow-posts",
        split="train",
        streaming=True,
    )

    for sample in ds:
        if len(accepted) >= max_collect:
            break

        scanned_pass1 += 1
        if scanned_pass1 % _LOG_EVERY == 0:
            logger.info(
                "stream_stackoverflow pass 1: %d scanned, %d collected",
                scanned_pass1,
                len(accepted),
            )

        if sample.get("PostTypeId") != 1:
            continue
        if (sample.get("Score") or 0) < 5:
            continue

        accepted_id = sample.get("AcceptedAnswerId")
        if accepted_id is None:
            continue

        tags = sample.get("Tags") or []
        if isinstance(tags, list):
            has_python = any("python" in t.lower() for t in tags)
        else:
            has_python = "python" in str(tags).lower()
        if not has_python:
            continue

        body = sample.get("Body") or ""
        if not body:
            continue

        accepted[accepted_id] = body

    logger.info(
        "stream_stackoverflow pass 1 done: %d scanned, %d questions collected",
        scanned_pass1,
        len(accepted),
    )

    if not accepted:
        logger.warning("stream_stackoverflow: no questions collected — yielding nothing")
        return

    logger.info("stream_stackoverflow pass 2: scanning for accepted answers")

    ds2 = load_dataset(
        "mikex86/stackoverflow-posts",
        split="train",
        streaming=True,
    )

    count = 0
    scanned_pass2 = 0
    for sample in ds2:
        if count >= max_docs:
            break

        scanned_pass2 += 1
        if scanned_pass2 % _LOG_EVERY == 0:
            logger.info(
                "stream_stackoverflow pass 2: %d scanned, %d yielded",
                scanned_pass2,
                count,
            )

        if sample.get("PostTypeId") != 2:
            continue

        answer_id = sample.get("Id")
        if answer_id not in accepted:
            continue

        answer_body = sample.get("Body") or ""
        if not answer_body:
            continue

        doc = f"Question:\n{accepted[answer_id]}\n\nAnswer:\n{answer_body}"
        yield doc
        count += 1

    logger.info(
        "stream_stackoverflow done: %d yielded from %d pass-2 rows scanned",
        count,
        scanned_pass2,
    )
