"""
data/download.py

Streams raw text from all four data sources. Writes nothing to disk.
Each function is a generator — yields one document at a time.

Sources:
    - The Stack v1 (Python)                          70% of final mix
    - Stack Overflow Python Q&A                      15%
    - Python docs + stdlib source (cpython/typeshed) 10%
    - Curated open source (NumPy, PyTorch, etc.)      5%

Confirmed field names (verified against live datasets):
    The Stack v1:   content, max_stars_repo_name, max_stars_count,
                    max_line_length, avg_line_length
    SO posts:       Id, PostTypeId, AcceptedAnswerId, ParentId,
                    Score, Body, Tags

Usage:
    from data.download import stream_stack, stream_stackoverflow, stream_docs, stream_curated

    for text in stream_stack(max_docs=1000):
        print(text[:100])
"""

import logging
from collections.abc import Iterator

logger = logging.getLogger(__name__)

_LOG_EVERY = 10_000


def stream_stack(max_docs: int = 500_000) -> Iterator[str]:
    """Stream Python source files from The Stack."""
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("pip install datasets") from err

    _excluded_repos = {
        "python/cpython",
        "python/typeshed",
        "numpy/numpy",
        "pytorch/pytorch",
        "pandas-dev/pandas",
        "tiangolo/fastapi",
        "psf/requests",
    }

    logger.info("stream_stack: starting (max_docs=%d)", max_docs)

    ds = load_dataset(
        "bigcode/the-stack",
        data_dir="data/python",
        split="train",
        streaming=True,
    )

    count = 0
    skipped = 0
    for sample in ds:
        repo = sample.get("max_stars_repo_name") or ""
        if repo in _excluded_repos:
            skipped += 1
            continue
        content = sample.get("content") or ""
        if not content:
            skipped += 1
            continue
        yield content
        count += 1
        if count % _LOG_EVERY == 0:
            logger.info("stream_stack: %d yielded, %d skipped", count, skipped)
        if count >= max_docs:
            break

    logger.info("stream_stack: done. %d yielded, %d skipped", count, skipped)


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

        post_type = sample.get("PostTypeId")
        if post_type != 1:
            continue

        score = sample.get("Score") or 0
        if score < 5:
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

        post_type = sample.get("PostTypeId")
        if post_type != 2:
            continue

        answer_id = sample.get("Id")
        if answer_id not in accepted:
            continue

        answer_body = sample.get("Body") or ""
        if not answer_body:
            continue

        question_body = accepted[answer_id]
        doc = f"Question:\n{question_body}\n\nAnswer:\n{answer_body}"
        yield doc
        count += 1

    logger.info(
        "stream_stackoverflow done. %d yielded from %d pass-2 rows scanned",
        count,
        scanned_pass2,
    )


def stream_docs(max_docs: int = 50_000) -> Iterator[str]:
    """Stream Python stdlib source from The Stack."""
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("pip install datasets") from err

    _stdlib_repos = {"python/cpython", "python/typeshed"}

    logger.info("stream_docs: starting (max_docs=%d)", max_docs)

    ds = load_dataset(
        "bigcode/the-stack",
        data_dir="data/python",
        split="train",
        streaming=True,
    )

    count = 0
    skipped = 0
    for sample in ds:
        repo = sample.get("max_stars_repo_name") or ""
        if repo not in _stdlib_repos:
            skipped += 1
            continue
        content = sample.get("content") or ""
        if not content:
            skipped += 1
            continue
        yield content
        count += 1
        if count % _LOG_EVERY == 0:
            logger.info("stream_docs: %d yielded, %d skipped", count, skipped)
        if count >= max_docs:
            break

    logger.info("stream_docs: done. %d yielded, %d skipped", count, skipped)


def stream_curated(max_docs: int = 30_000) -> Iterator[str]:
    """Stream source code from idiomatic Python projects."""
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("pip install datasets") from err

    _curated_repos = {
        "numpy/numpy",
        "pytorch/pytorch",
        "pandas-dev/pandas",
        "tiangolo/fastapi",
        "psf/requests",
    }

    logger.info("stream_curated: starting (max_docs=%d)", max_docs)

    ds = load_dataset(
        "bigcode/the-stack",
        data_dir="data/python",
        split="train",
        streaming=True,
    )

    count = 0
    skipped = 0
    for sample in ds:
        repo = sample.get("max_stars_repo_name") or ""
        if repo not in _curated_repos:
            skipped += 1
            continue
        content = sample.get("content") or ""
        if not content:
            skipped += 1
            continue
        yield content
        count += 1
        if count % _LOG_EVERY == 0:
            logger.info("stream_curated: %d yielded, %d skipped", count, skipped)
        if count >= max_docs:
            break

    logger.info("stream_curated: done. %d yielded, %d skipped", count, skipped)
