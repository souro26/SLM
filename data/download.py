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

from collections.abc import Iterator


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

    ds = load_dataset(
        "bigcode/the-stack",
        data_dir="data/python",
        split="train",
        streaming=True,
    )

    count = 0
    for sample in ds:
        repo = sample.get("max_stars_repo_name") or ""
        if repo in _excluded_repos:
            continue
        content = sample.get("content") or ""
        if not content:
            continue
        yield content
        count += 1
        if count >= max_docs:
            break


def stream_stackoverflow(max_docs: int = 200_000) -> Iterator[str]:
    """Stream Python Q&A pairs from Stack Overflow."""
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("pip install datasets") from err

    max_collect = max_docs * 5

    accepted: dict[int, str] = {}

    ds = load_dataset(
        "mikex86/stackoverflow-posts",
        split="train",
        streaming=True,
    )

    for sample in ds:
        if len(accepted) >= max_collect:
            break

        post_type = sample.get("PostTypeId")
        if post_type != 1:  # questions only
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

    if not accepted:
        return

    ds2 = load_dataset(
        "mikex86/stackoverflow-posts",
        split="train",
        streaming=True,
    )

    count = 0
    for sample in ds2:
        if count >= max_docs:
            break

        post_type = sample.get("PostTypeId")
        if post_type != 2:  # answers only
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


def stream_docs(max_docs: int = 50_000) -> Iterator[str]:
    """Stream Python stdlib source from The Stack."""
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("pip install datasets") from err

    _stdlib_repos = {"python/cpython", "python/typeshed"}

    ds = load_dataset(
        "bigcode/the-stack",
        data_dir="data/python",
        split="train",
        streaming=True,
    )

    count = 0
    for sample in ds:
        repo = sample.get("max_stars_repo_name") or ""
        if repo not in _stdlib_repos:
            continue
        content = sample.get("content") or ""
        if not content:
            continue
        yield content
        count += 1
        if count >= max_docs:
            break


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

    ds = load_dataset(
        "bigcode/the-stack",
        data_dir="data/python",
        split="train",
        streaming=True,
    )

    count = 0
    for sample in ds:
        repo = sample.get("max_stars_repo_name") or ""
        if repo not in _curated_repos:
            continue
        content = sample.get("content") or ""
        if not content:
            continue
        yield content
        count += 1
        if count >= max_docs:
            break
