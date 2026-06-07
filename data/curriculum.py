"""
data/curriculum.py

Sorts deduplicated Python files into three training stages by complexity.
The model sees simple code first, then complex code, then the best code last.

Why curriculum learning:
    A model trained on randomly shuffled data has to simultaneously learn basic
    syntax AND complex patterns like async, decorators, metaclasses. Starting
    with simple, well-structured code lets the model build syntactic grounding
    first. Complex patterns are introduced once the basics are stable.

    Stage 3 (cooldown) upsamples the highest quality data at the end of
    training — this is standard practice, same approach used in LLaMA 3.

Stages:
    Stage 1 — Simple code. Short functions, clear names, docstrings present.
              Low cyclomatic complexity, few nested scopes.
    Stage 2 — Complex code. Classes, decorators, generators, async, context
              managers. Higher complexity, more unique identifiers.
    Stage 3 — Cooldown. Best quality data only — curated repos, stdlib,
              top Stack Overflow answers. Upsampled 3x at end of training.

Complexity measured by:
    - Cyclomatic complexity (radon cc_visit, falls back to AST branch count)
    - Number of nested scopes (AST walk)
    - Number of unique identifiers (AST walk)

Usage:
    from data.curriculum import score_complexity, assign_stage, split_by_stage

    stage1, stage2, stage3 = split_by_stage(deduplicated_stream)
"""

import ast
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

try:
    from radon.complexity import cc_visit

    HAS_RADON = True
except ImportError:
    HAS_RADON = False
    logger.warning(
        "radon not installed — falling back to AST branch count for cyclomatic complexity. "
        "Install with: pip install radon"
    )

_LOG_EVERY = 10_000

Stage = Literal[1, 2, 3]

STAGE1_MAX_SCORE = 5.0
STAGE2_MAX_SCORE = 15.0
STAGE3_UPSAMPLE = 3
STAGE3_SOURCES = {"curated", "stdlib", "docs"}


@dataclass
class ComplexityScore:
    cyclomatic: float
    max_nesting: int
    unique_identifiers: int
    composite: float


def _cyclomatic_complexity(source: str) -> float:
    """Average cyclomatic complexity across all functions in the file."""
    if HAS_RADON:
        try:
            results = cc_visit(source)
            if not results:
                return 1.0
            return sum(r.complexity for r in results) / len(results)
        except Exception:
            return 1.0

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return 1.0

    branch_nodes = (
        ast.If,
        ast.For,
        ast.While,
        ast.ExceptHandler,
        ast.With,
        ast.Assert,
        ast.comprehension,
    )
    branches = sum(1 for node in ast.walk(tree) if isinstance(node, branch_nodes))
    functions = sum(
        1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    )
    if functions == 0:
        return 1.0 + branches * 0.5
    return 1.0 + branches / functions


def _max_nesting_depth(source: str) -> int:
    """Maximum nesting depth of scopes in the file."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return 0

    scope_nodes = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.If,
        ast.For,
        ast.While,
        ast.With,
        ast.Try,
    )

    def _depth(node: ast.AST, current: int) -> int:
        max_d = current
        for child in ast.iter_child_nodes(node):
            if isinstance(child, scope_nodes):
                max_d = max(max_d, _depth(child, current + 1))
            else:
                max_d = max(max_d, _depth(child, current))
        return max_d

    return _depth(tree, 0)


def _unique_identifiers(source: str) -> int:
    """Number of distinct names (variables, functions, classes) in the file."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return 0
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    return len(names)


def score_complexity(source: str) -> ComplexityScore:
    """Compute a composite complexity score for a source file."""
    cyclomatic = _cyclomatic_complexity(source)
    nesting = _max_nesting_depth(source)
    identifiers = _unique_identifiers(source)
    composite = 0.5 * cyclomatic + 0.3 * nesting + 0.2 * (identifiers / 20)

    return ComplexityScore(
        cyclomatic=cyclomatic,
        max_nesting=nesting,
        unique_identifiers=identifiers,
        composite=composite,
    )


def assign_stage(source: str, source_tag: str = "") -> Stage:
    """Assign a training stage to a source file."""
    if source_tag in STAGE3_SOURCES:
        return 3

    score = score_complexity(source)

    if score.composite <= STAGE1_MAX_SCORE:
        return 1
    else:
        return 2


def split_by_stage(
    source_iter: Iterator[tuple[str, str]],
) -> tuple[list[str], list[str], list[str]]:
    """Split a deduplicated stream into three stage lists."""
    stage1: list[str] = []
    stage2: list[str] = []
    stage3: list[str] = []
    total = 0

    for source, source_tag in source_iter:
        stage = assign_stage(source, source_tag)
        total += 1

        if stage == 1:
            stage1.append(source)
        elif stage == 2:
            stage2.append(source)
        else:
            stage3.append(source)

        if total % _LOG_EVERY == 0:
            logger.info(
                "curriculum: %d processed — stage1=%d, stage2=%d, stage3=%d",
                total,
                len(stage1),
                len(stage2),
                len(stage3),
            )

    logger.info(
        "curriculum done: %d files — stage1=%d (%.1f%%), stage2=%d (%.1f%%), stage3=%d (%.1f%%)",
        total,
        len(stage1),
        len(stage1) / max(total, 1) * 100,
        len(stage2),
        len(stage2) / max(total, 1) * 100,
        len(stage3),
        len(stage3) / max(total, 1) * 100,
    )

    return stage1, stage2, stage3


def ordered_stream(
    stage1: list[str],
    stage2: list[str],
    stage3: list[str],
) -> Iterator[str]:
    """Yield files in curriculum order: Stage 1 → Stage 2 → Stage 3."""
    logger.info(
        "ordered_stream: yielding %d stage1, %d stage2, %d stage3 (×%d)",
        len(stage1),
        len(stage2),
        len(stage3),
        STAGE3_UPSAMPLE,
    )
    yield from stage1
    yield from stage2
    for i in range(STAGE3_UPSAMPLE):
        logger.info("ordered_stream: stage3 pass %d / %d", i + 1, STAGE3_UPSAMPLE)
        yield from stage3
