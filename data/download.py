"""
data/download.py

Streams raw text from all four data sources. Writes nothing to disk.
Each function is a generator — yields one document at a time.

Sources:
    - The Stack v2 (Python)                          70% of final mix
    - Stack Overflow Python Q&A                      15%
    - Python docs + stdlib source                    10%
    - Curated open source (NumPy, PyTorch, etc.)      5%

Usage:
    from data.download import stream_stack, stream_stackoverflow, stream_docs, stream_curated

    for text in stream_stack(max_docs=10000):
        print(text[:100])
"""

import re
from collections.abc import Iterator


def stream_stack(max_docs: int = 500_000) -> Iterator[str]:
    """Stream Python source files from The Stack v2 (HuggingFace)."""
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("pip install datasets") from err

    ds = load_dataset(
        "bigcode/the-stack-v2",
        "Python",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    count = 0
    for sample in ds:
        content = sample.get("content", "") or sample.get("text", "")
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

    ds = load_dataset(
        "keirp/stackoverflow-python-dataset",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    count = 0
    for sample in ds:
        question_score = sample.get("question_score", 0) or 0
        if question_score < 5:
            continue

        accepted_answer = sample.get("accepted_answer", "") or ""
        if not accepted_answer:
            continue

        question_body = _strip_html(sample.get("question_body", "") or "")
        answer_body = _strip_html(accepted_answer)

        if not question_body or not answer_body:
            continue

        doc = f"Question: {question_body}\n\nAnswer:\n{answer_body}"
        yield doc
        count += 1
        if count >= max_docs:
            break


def stream_docs(max_docs: int = 50_000) -> Iterator[str]:
    """Stream Python documentation and stdlib source code."""
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("pip install datasets") from err

    count = 0

    ds = load_dataset(
        "bigcode/the-stack-v2",
        "Python",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    stdlib_repos = {"python/cpython", "python/typeshed"}

    for sample in ds:
        repo = sample.get("repo_name", "") or ""
        if repo not in stdlib_repos:
            continue
        content = sample.get("content", "") or sample.get("text", "")
        if not content:
            continue
        yield content
        count += 1
        if count >= max_docs:
            return

    try:
        docs_ds = load_dataset(
            "bigcode/the-stack-v2-dedup",
            "Python",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        for sample in docs_ds:
            content = sample.get("content", "") or ""
            if not content:
                continue
            yield content
            count += 1
            if count >= max_docs:
                return
    except Exception:
        pass


def stream_curated(max_docs: int = 30_000) -> Iterator[str]:
    """Stream source code from high-quality, idiomatic Python projects."""
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("pip install datasets") from err

    curated_repos = {
        "numpy/numpy",
        "pytorch/pytorch",
        "pandas-dev/pandas",
        "tiangolo/fastapi",
        "psf/requests",
    }

    ds = load_dataset(
        "bigcode/the-stack-v2",
        "Python",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    count = 0
    for sample in ds:
        repo = sample.get("repo_name", "") or ""
        if repo not in curated_repos:
            continue
        content = sample.get("content", "") or sample.get("text", "")
        if not content:
            continue
        yield content
        count += 1
        if count >= max_docs:
            break


def _strip_html(text: str) -> str:
    """Strip HTML tags from Stack Overflow post bodies."""
    if not text:
        return ""

    text = re.sub(
        r"<pre[^>]*><code[^>]*>(.*?)</code></pre>",
        lambda m: "\n" + _unescape_html(m.group(1)) + "\n",
        text,
        flags=re.DOTALL,
    )

    text = re.sub(
        r"<code[^>]*>(.*?)</code>",
        lambda m: _unescape_html(m.group(1)),
        text,
        flags=re.DOTALL,
    )

    text = re.sub(r"<[^>]+>", " ", text)
    text = _unescape_html(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def _unescape_html(text: str) -> str:
    """Unescape common HTML entities."""
    entities = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&nbsp;": " ",
        "&#x27;": "'",
        "&#x2F;": "/",
    }
    for entity, char in entities.items():
        text = text.replace(entity, char)
    return text
