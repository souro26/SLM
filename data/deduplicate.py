"""
data/deduplicate.py

MinHash deduplication at the file level.
Removes near-duplicate files — cross-fork copies, copy-pasted code,
files that appear in multiple repos with minor modifications.

Why deduplication matters:
    Without dedup, the model sees the same file hundreds of times across forks.
    It memorizes those files instead of learning general patterns.
    Dedup also prevents eval set contamination — if a HumanEval problem solution
    appears verbatim in training data, pass@1 is meaningless.

Algorithm:
    MinHash estimates Jaccard similarity between documents without comparing
    them directly. Each document is represented as a fixed-size signature
    (128 hash values). Two documents are near-duplicates if their signature
    similarity exceeds a threshold (0.85).

    We use shingling — the document is represented as a set of overlapping
    n-grams of tokens. Character 5-grams work well for code.

Usage:
    from data.deduplicate import Deduplicator

    dedup = Deduplicator(threshold=0.85)
    for text in dedup.filter_stream(filtered_stream):
        # text is not a near-duplicate of anything seen before
        ...
"""

import logging
import re
import unicodedata
from collections.abc import Iterator

from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)

NUM_PERM = 128
THRESHOLD = 0.85
SHINGLE_SIZE = 5

_LOG_EVERY = 10_000


def normalize(source: str) -> str:
    """Normalize source code before shingling."""
    source = unicodedata.normalize("NFC", source)
    source = re.sub(r"[^\S\n]+", " ", source)
    return source.strip()


def shingle(source: str, size: int = SHINGLE_SIZE) -> set:
    """Convert source code to a set of character n-grams (shingles)."""
    source = normalize(source)
    if len(source) < size:
        return {source}
    return {source[i : i + size] for i in range(len(source) - size + 1)}


def make_minhash(source: str) -> MinHash:
    """Compute MinHash signature for a source file."""
    m = MinHash(num_perm=NUM_PERM)
    for shingle_str in shingle(source):
        m.update(shingle_str.encode("utf-8"))
    return m


class Deduplicator:

    def __init__(self, threshold: float = THRESHOLD, num_perm: int = NUM_PERM):
        self.threshold = threshold
        self.num_perm = num_perm
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._seen = 0
        self._duplicates = 0
        logger.info(
            "Deduplicator initialised (threshold=%.2f, num_perm=%d)",
            threshold,
            num_perm,
        )

    def is_unique(self, source: str) -> bool:
        """Check if source is a near-duplicate of anything seen so far."""
        m = make_minhash(source)
        self._seen += 1

        if self._lsh.query(m):
            self._duplicates += 1
            return False

        self._lsh.insert(str(self._seen), m)
        return True

    def filter_stream(
        self,
        source_iter: Iterator[str],
    ) -> Iterator[str]:
        """Apply deduplication to a stream of source files."""
        for source in source_iter:
            is_unique = self.is_unique(source)

            if self._seen % _LOG_EVERY == 0:
                dup_rate = self._duplicates / self._seen * 100
                logger.info(
                    "dedup: %d seen, %d duplicates removed (%.1f%%)",
                    self._seen,
                    self._duplicates,
                    dup_rate,
                )

            if is_unique:
                yield source

        logger.info(
            "dedup done: %d unique / %d total (%d duplicates removed, %.1f%%)",
            self._seen - self._duplicates,
            self._seen,
            self._duplicates,
            self._duplicates / max(self._seen, 1) * 100,
        )

    @property
    def stats(self) -> dict:
        return {
            "seen": self._seen,
            "duplicates": self._duplicates,
            "unique": self._seen - self._duplicates,
            "duplicate_rate": self._duplicates / max(self._seen, 1),
        }
