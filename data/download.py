"""
data/download.py  (v3)

Streams raw text from all four data sources. Writes nothing to disk
except transient repo tarballs (cleaned up automatically).

Sources:
    - The Stack v1 (Python, general sample)          70% of final mix
    - Stack Overflow Python Q&A                       15%
    - Python docs + stdlib source (cpython/typeshed)  10%
    - Curated open source (NumPy, PyTorch, etc.)       5%

Key design decision (v3):
    v1 had three separate full scans over The Stack (3x redundant downloads).
    v2 collapsed this to a single pass, but the single pass still had to
    scan deep into the dataset hunting for files from 7 specific repos
    (5 curated + 2 stdlib) out of millions — the loop couldn't exit until
    ALL THREE counters were full, so it was still bottlenecked by the
    rarest category. Fixing "3 passes" to "1 pass" didn't fix "needle in
    haystack."

    v3 fix: stop searching for known repos inside The Stack at all.
    stream_stack_all() now ONLY samples the general population and exits
    as soon as stack_limit is hit — nothing else to wait for. Curated and
    stdlib repos are fetched DIRECTLY from GitHub via tarball download
    (fetch_curated_repos / fetch_stdlib_repos) since we already know
    exactly which 7 repos we want. No scanning, no waiting on rare finds.

Confirmed field names (verified against live datasets):
    The Stack v1:   content, max_stars_repo_name, max_stars_count,
                    max_line_length, avg_line_length
    SO posts:       Id, PostTypeId, AcceptedAnswerId, ParentId,
                    Score, Body, Tags

Usage:
    from data.download import stream_stack_all, stream_stackoverflow
    from data.download import fetch_curated_repos, fetch_stdlib_repos

    for text in stream_stack_all(stack_limit=5000):
        ...
    for text in fetch_curated_repos(limit=300):
        ...
    for text in fetch_stdlib_repos(limit=500):
        ...
"""

import io
import logging
import tarfile
import tempfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

_LOG_EVERY = 10_000

_CURATED_REPOS = [
    "numpy/numpy",
    "pytorch/pytorch",
    "pandas-dev/pandas",
    "tiangolo/fastapi",
    "psf/requests",
]

_STDLIB_REPOS = [
    "python/cpython",
    "python/typeshed",
]

_TARBALL_URL = "https://codeload.github.com/{repo}/tar.gz/refs/heads/{branch}"
_DEFAULT_BRANCHES = ("main", "master")


def _fetch_repo_py_files(repo: str, per_repo_limit: int) -> Iterator[str]:
    """Download a repo's tarball from GitHub and yield contents of .py files."""
    last_err = None
    for branch in _DEFAULT_BRANCHES:
        url = _TARBALL_URL.format(repo=repo, branch=branch)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            break
        except Exception as err:  # noqa
            last_err = err
            data = None
    else:
        logger.warning("fetch failed for %s: %s", repo, last_err)
        return

    yielded = 0
    with tempfile.TemporaryDirectory() as tmp:
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
                tf.extractall(tmp, filter="data")
        except TypeError:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
                tf.extractall(tmp)

        for path in Path(tmp).rglob("*.py"):
            if yielded >= per_repo_limit:
                break
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not content.strip():
                continue
            yield content
            yielded += 1

    logger.info("fetched %s: %d .py files", repo, yielded)


def fetch_curated_repos(limit: int = 300) -> Iterator[str]:
    """Fetch .py files directly from curated open-source repos on GitHub.

    No scanning of The Stack — these repos are known in advance, so we
    just clone/download them directly. Fast and not dependent on how
    rare they are inside a large streamed dataset.
    """
    per_repo = max(1, limit // len(_CURATED_REPOS))
    logger.info(
        "fetch_curated_repos: starting (limit=%d, per_repo=%d, repos=%s)",
        limit,
        per_repo,
        _CURATED_REPOS,
    )
    count = 0
    for repo in _CURATED_REPOS:
        if count >= limit:
            break
        for content in _fetch_repo_py_files(repo, per_repo_limit=per_repo):
            if count >= limit:
                break
            yield content
            count += 1
    logger.info("fetch_curated_repos done: %d files", count)


def fetch_stdlib_repos(limit: int = 500) -> Iterator[str]:
    """Fetch .py files directly from cpython/typeshed on GitHub."""
    per_repo = max(1, limit // len(_STDLIB_REPOS))
    logger.info(
        "fetch_stdlib_repos: starting (limit=%d, per_repo=%d, repos=%s)",
        limit,
        per_repo,
        _STDLIB_REPOS,
    )
    count = 0
    for repo in _STDLIB_REPOS:
        if count >= limit:
            break
        for content in _fetch_repo_py_files(repo, per_repo_limit=per_repo):
            if count >= limit:
                break
            yield content
            count += 1
    logger.info("fetch_stdlib_repos done: %d files", count)


def stream_stack_all(stack_limit: int = 500_000) -> Iterator[str]:
    """Single-pass, single-purpose stream over The Stack v1.

    v3: ONLY samples the general population up to stack_limit. Curated
    and stdlib repos are no longer hunted for here (see fetch_curated_repos
    / fetch_stdlib_repos) — this loop exits as soon as stack_limit is hit,
    with no rare-repo condition left to wait on.

    Files belonging to the curated/stdlib repos are skipped here (not
    yielded) to avoid duplicate content, since those repos are already
    covered by the direct GitHub fetch.
    """
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("pip install datasets") from err

    logger.info("stream_stack_all: starting (stack_limit=%d)", stack_limit)

    skip_repos = set(_CURATED_REPOS) | set(_STDLIB_REPOS)

    ds = load_dataset(
        "bigcode/the-stack",
        data_dir="data/python",
        split="train",
        streaming=True,
    )

    stack_count = 0
    scanned = 0
    skipped = 0

    for sample in ds:
        if stack_count >= stack_limit:
            break

        scanned += 1

        content = sample.get("content") or ""
        if not content:
            skipped += 1
            continue

        repo = sample.get("max_stars_repo_name") or ""
        if repo in skip_repos:
            continue

        stack_count += 1
        yield content

        if scanned % _LOG_EVERY == 0:
            logger.info(
                "stream_stack_all: %d scanned — stack=%d, skipped=%d",
                scanned,
                stack_count,
                skipped,
            )

    logger.info(
        "stream_stack_all done: %d scanned — stack=%d, skipped=%d",
        scanned,
        stack_count,
        skipped,
    )


def stream_stackoverflow(max_docs: int = 200_000) -> Iterator[str]:
    """Stream Python Q&A pairs from Stack Overflow. Unchanged from v2."""
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
