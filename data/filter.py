"""
data/filter.py

Quality filters for Python source files.
Applied to all Stack v1 files. Stack Overflow filtering is
already handled in download.py — don't run these filters on SO docs.

Filters applied in order (cheapest first):
    1. Minimum length         — skip near-empty files
    2. Line length            — filter minified / generated code
    3. Comment-to-code ratio  — filter undocumented garbage and comment-only files
    4. ast.parse() validity   — hard filter, discard broken syntax (expensive, runs once)
    5. Docstring presence     — proxy for intentional, well-written code (reuses parse tree)

Note on ordering: ast.parse() is the most expensive operation. All cheap
string-based checks run first so broken/garbage files are discarded before
we pay the cost of parsing. Once we parse, we reuse the tree for the
docstring check — we never parse the same file twice.

v2 change: _parse() now catches RecursionError and MemoryError in addition
to SyntaxError/ValueError. A small number of files in The Stack have
pathologically deep nesting (huge nested literals, deeply chained
expressions, generated code) that blow CPython's AST-construction
recursion limit. That's a RecursionError, not a SyntaxError, and it was
previously uncaught — one such file would crash the entire pipeline run
after hours of otherwise-successful work. Now it's just treated as a
file that fails the syntax filter, same as any other unparseable file.

Usage:
    from data.filter import filter_stream

    for text in filter_stream(stream_stack()):
        # text passed all filters
        ...
"""

import ast
import logging
from collections.abc import Iterator

logger = logging.getLogger(__name__)

MIN_CHARS = 100
MIN_LINES = 5
MAX_LINE_LENGTH = 1000
MAX_LONG_LINE_FRACTION = 0.2
MIN_COMMENT_RATIO = 0.01
MAX_COMMENT_RATIO = 0.80

_LOG_EVERY = 10_000

# Exceptions ast.parse() can raise on pathological (but not necessarily
# "syntactically invalid" in the SyntaxError sense) input. Any of these
# mean "this file can't be parsed" — treat it as a filtered-out file,
# never let it propagate and kill the pipeline.
_PARSE_FAILURE_EXCEPTIONS = (SyntaxError, ValueError, RecursionError, MemoryError)


def is_long_enough(source: str) -> bool:
    """Skip near-empty files."""
    if len(source) < MIN_CHARS:
        return False
    if source.count("\n") < MIN_LINES:
        return False
    return True


def has_acceptable_line_lengths(source: str) -> bool:
    """Filter minified or generated code."""
    lines = source.splitlines()
    if not lines:
        return False
    long_lines = sum(1 for line in lines if len(line) > MAX_LINE_LENGTH)
    fraction = long_lines / len(lines)
    return fraction <= MAX_LONG_LINE_FRACTION


def has_acceptable_comment_ratio(source: str) -> bool:
    """Filter files by comment ratio."""
    lines = source.splitlines()
    non_blank = [line for line in lines if line.strip()]
    if not non_blank:
        return False
    comment_lines = sum(1 for line in non_blank if line.strip().startswith("#"))
    ratio = comment_lines / len(non_blank)
    return MIN_COMMENT_RATIO <= ratio <= MAX_COMMENT_RATIO


def _parse(source: str) -> ast.Module | None:
    """Parse source into an AST.

    Returns None for anything unparseable — genuine syntax errors,
    pathologically deep nesting (RecursionError), or oversized/degenerate
    input (MemoryError). None of these should ever propagate up and
    kill the pipeline; a single bad file just gets filtered out.
    """
    try:
        return ast.parse(source)
    except _PARSE_FAILURE_EXCEPTIONS:
        return None


def has_valid_syntax(source: str) -> bool:
    """Discard files that fail ast.parse()."""
    return _parse(source) is not None


def has_docstring(tree: ast.Module) -> bool:
    """Check if file contains a docstring."""
    if ast.get_docstring(tree):
        return True

    for node in ast.walk(tree):
        if isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ) and ast.get_docstring(node):
            return True

    return False


def is_quality_file(
    source: str,
    require_docstring: bool = True,
) -> bool:
    """Run all filters on a source file."""
    if not is_long_enough(source):
        return False

    if not has_acceptable_line_lengths(source):
        return False

    if not has_acceptable_comment_ratio(source):
        return False

    tree = _parse(source)
    if tree is None:
        return False

    if require_docstring and not has_docstring(tree):
        return False

    return True


def filter_stream(
    source_iter: Iterator[str],
    require_docstring: bool = True,
) -> Iterator[str]:
    """Apply quality filters to a source stream."""
    total = 0
    passed = 0

    rejected_length = 0
    rejected_line_length = 0
    rejected_comment_ratio = 0
    rejected_syntax = 0
    rejected_docstring = 0

    for source in source_iter:
        total += 1

        if not is_long_enough(source):
            rejected_length += 1
            if total % _LOG_EVERY == 0:
                _log_progress(
                    total,
                    passed,
                    rejected_length,
                    rejected_line_length,
                    rejected_comment_ratio,
                    rejected_syntax,
                    rejected_docstring,
                )
            continue

        if not has_acceptable_line_lengths(source):
            rejected_line_length += 1
            if total % _LOG_EVERY == 0:
                _log_progress(
                    total,
                    passed,
                    rejected_length,
                    rejected_line_length,
                    rejected_comment_ratio,
                    rejected_syntax,
                    rejected_docstring,
                )
            continue

        if not has_acceptable_comment_ratio(source):
            rejected_comment_ratio += 1
            if total % _LOG_EVERY == 0:
                _log_progress(
                    total,
                    passed,
                    rejected_length,
                    rejected_line_length,
                    rejected_comment_ratio,
                    rejected_syntax,
                    rejected_docstring,
                )
            continue

        tree = _parse(source)
        if tree is None:
            rejected_syntax += 1
            if total % _LOG_EVERY == 0:
                _log_progress(
                    total,
                    passed,
                    rejected_length,
                    rejected_line_length,
                    rejected_comment_ratio,
                    rejected_syntax,
                    rejected_docstring,
                )
            continue

        if require_docstring and not has_docstring(tree):
            rejected_docstring += 1
            if total % _LOG_EVERY == 0:
                _log_progress(
                    total,
                    passed,
                    rejected_length,
                    rejected_line_length,
                    rejected_comment_ratio,
                    rejected_syntax,
                    rejected_docstring,
                )
            continue

        passed += 1
        if total % _LOG_EVERY == 0:
            _log_progress(
                total,
                passed,
                rejected_length,
                rejected_line_length,
                rejected_comment_ratio,
                rejected_syntax,
                rejected_docstring,
            )
        yield source

    logger.info(
        "filter done: %d processed, %d passed (%.1f%%) | "
        "rejected: length=%d line_length=%d comment_ratio=%d syntax=%d docstring=%d",
        total,
        passed,
        passed / max(total, 1) * 100,
        rejected_length,
        rejected_line_length,
        rejected_comment_ratio,
        rejected_syntax,
        rejected_docstring,
    )


def _log_progress(
    total: int,
    passed: int,
    rejected_length: int,
    rejected_line_length: int,
    rejected_comment_ratio: int,
    rejected_syntax: int,
    rejected_docstring: int,
) -> None:
    """Log a progress line. Called every _LOG_EVERY files."""
    logger.info(
        "filter: %d processed, %d passed (%.1f%%) | "
        "rejected: length=%d line_len=%d comment=%d syntax=%d docstring=%d",
        total,
        passed,
        passed / max(total, 1) * 100,
        rejected_length,
        rejected_line_length,
        rejected_comment_ratio,
        rejected_syntax,
        rejected_docstring,
    )
