"""
tests/performance/test_tokenizer_perf.py

Performance tests for SLMTokenizer.
Run separately — not part of the default test suite.

    pytest tests/performance/ -v
"""

import time
from pathlib import Path

import pytest

from tokenizer.tokenizer import SLMTokenizer

TOKENIZER_DIR = Path("tokenizer/trained")


@pytest.fixture(scope="module")
def tok():
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        pytest.skip("Trained tokenizer not found. Run tokenizer/train.py first.")
    return SLMTokenizer(TOKENIZER_DIR)


PERF_SAMPLE = (
    """
import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.o_proj(v)
"""
    * 10
)


def test_single_encode_latency(tok):
    """Single sequence encode must complete in under 100ms."""
    text = PERF_SAMPLE[:2000]
    tok.encode(text)
    t0 = time.perf_counter()
    tok.encode(text)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 100, f"Single encode took {elapsed_ms:.1f}ms, expected < 100ms"


def test_batch_encode_latency(tok):
    """Batch of 32 sequences must encode in under 500ms."""
    texts = [PERF_SAMPLE[:500]] * 32
    tok.encode_batch(texts)
    t0 = time.perf_counter()
    tok.encode_batch(texts)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 500, f"Batch encode took {elapsed_ms:.1f}ms, expected < 500ms"


def test_throughput_tokens_per_second(tok):
    """Must sustain at least 500k tokens/sec."""
    text = PERF_SAMPLE
    ids = tok.encode(text)
    n_repeats = 50
    t0 = time.perf_counter()
    total_tokens = 0
    for _ in range(n_repeats):
        ids = tok.encode(text)
        total_tokens += len(ids)
    elapsed = time.perf_counter() - t0
    tokens_per_sec = total_tokens / elapsed
    assert (
        tokens_per_sec > 500_000
    ), f"Throughput too low: {tokens_per_sec:,.0f} tokens/sec (expected > 500k)"


def test_context_length_boundary(tok):
    """Encoding exactly 2048 tokens must work without error."""
    text = "x = 1\n" * 1000
    ids = tok.encode(text)
    ids_2048 = ids[:2048]
    assert len(ids_2048) == 2048
    decoded = tok.decode(ids_2048)
    assert isinstance(decoded, str)
    assert len(decoded) > 0


def test_context_length_exceeded(tok):
    """Encoding beyond 2048 tokens must not crash — tokenizer has no hard limit."""
    text = "x = 1\n" * 5000
    ids = tok.encode(text)
    assert len(ids) > 2048
    decoded = tok.decode(ids)
    assert isinstance(decoded, str)


def test_encode_throughput_large_file(tok):
    """Single large file (100k chars) must encode in under 200ms."""
    text = "x = 1\n" * 20_000
    t0 = time.perf_counter()
    ids = tok.encode(text)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert len(ids) > 0
    assert elapsed_ms < 200, f"Large file encode took {elapsed_ms:.1f}ms, expected < 200ms"
