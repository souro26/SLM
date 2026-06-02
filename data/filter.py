"""
data/filter.py

Quality filters for Python source files.
All filters are applied to The Stack v2 files.
Stack Overflow filtering is already handled in download.py.

Filters applied in order (cheapest first):
    1. Minimum length         — skip near-empty files
    2. ast.parse() validity   — hard filter, discard broken syntax
    3. Line length            — filter minified code
    4. Comment-to-code ratio  — filter undocumented garbage and comment-only files
    5. Docstring presence     — proxy for intentional, well-written code
    6. Minimum stars          — proxy for repo quality (if metadata available)

Usage:
    from data.filter import is_quality_file, filter_stream

    for text in filter_stream(stream_stack()):
        # text passed all filters
        ...
""" """
data/filter.py

Quality filters for Python source files.
All filters are applied to The Stack v2 files.
Stack Overflow filtering is already handled in download.py.

Filters applied in order (cheapest first):
    1. Minimum length         — skip near-empty files
    2. ast.parse() validity   — hard filter, discard broken syntax
    3. Line length            — filter minified code
    4. Comment-to-code ratio  — filter undocumented garbage and comment-only files
    5. Docstring presence     — proxy for intentional, well-written code
    6. Minimum stars          — proxy for repo quality (if metadata available)

Usage:
    from data.filter import is_quality_file, filter_stream

    for text in filter_stream(stream_stack()):
        # text passed all filters
        ...
"""

import ast
from collections.abc import Iterator

MIN_CHARS = 100
MIN_LINES = 5
MAX_LINE_LENGTH = 1000
MAX_LONG_LINE_FRACTION = 0.2
MIN_COMMENT_RATIO = 0.01
MAX_COMMENT_RATIO = 0.8
MIN_STARS = 1


def is_long_enough(source: str) -> bool:
    """Skip near-empty files."""
    if len(source) < MIN_CHARS:
        return False
    if source.count("\n") < MIN_LINES:
        return False
    return True


def is_valid_syntax(source: str) -> bool:
    """Discard files that fail ast.parse()."""
    try:
        ast.parse(source)
        return True
    except (SyntaxError, ValueError):
        return False


def has_acceptable_line_lengths(source: str) -> bool:
    """Filter out undocumented code or only comment files."""
    lines = source.splitlines()
    if not lines:
        return False

    long_lines = sum(1 for line in lines if len(line) > MAX_LINE_LENGTH)
    fraction = long_lines / len(lines)
    return fraction <= MAX_LONG_LINE_FRACTION


def has_acceptable_comment_ratio(source: str) -> bool:
    """Filter out undocumented code or the comment only files."""
    lines = source.splitlines()
    non_blank = [line for line in lines if line.strip()]
    if not non_blank:
        return False

    comment_lines = sum(1 for line in non_blank if line.strip().startswith("#"))
    ratio = comment_lines / len(non_blank)
    return MIN_COMMENT_RATIO <= ratio <= MAX_COMMENT_RATIO


def has_docstring(source: str) -> bool:
    """Check if the file has atleast one docstring."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False

    if ast.get_docstring(tree):
        return True

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            return True

    return False


def has_enough_stars(stars: int | None) -> bool:
    """Filter for number of stars."""
    if stars is None:
        return True
    return stars >= MIN_STARS


def is_quality_file(
    source: str,
    stars: int | None = None,
    require_docstring: bool = True,
) -> bool:
    """Run all filters on a single source file."""
    if not is_long_enough(source):
        return False

    if not has_acceptable_line_lengths(source):
        return False

    if not has_acceptable_comment_ratio(source):
        return False

    if not has_enough_stars(stars):
        return False

    if require_docstring and not has_docstring(source):
        return False

    if not is_valid_syntax(source):
        return False

    return True


def filter_stream(
    source_iter: Iterator[str],
    stars_iter: Iterator[int | None] | None = None,
    require_docstring: bool = True,
    log_every: int = 10_000,
) -> Iterator[str]:
    """Apply quality filters to a stream of source files."""
    total = 0
    passed = 0

    for source in source_iter:
        stars = next(stars_iter) if stars_iter else None
        total += 1

        if is_quality_file(source, stars=stars, require_docstring=require_docstring):
            passed += 1
            yield source

        if total % log_every == 0:
            rate = passed / total * 100
            print(f"  filter: {total:,} processed, {passed:,} passed ({rate:.1f}%)")

    print(f"Filter done. {passed:,} / {total:,} files passed ({passed/max(total,1)*100:.1f}%)")
