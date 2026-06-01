"""
tests/test_tokenizer.py

Tests for the SLMTokenizer wrapper.
Requires a trained tokenizer at tokenizer/trained/.
Run tokenizer/train.py first, then: pytest tests/test_tokenizer.py
"""

from pathlib import Path

import pytest

from tokenizer.tokenizer import SLMTokenizer

TOKENIZER_DIR = Path("tokenizer/trained")


@pytest.fixture(scope="module")
def tok():
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        pytest.skip("Trained tokenizer not found. Run tokenizer/train.py first.")
    return SLMTokenizer(TOKENIZER_DIR)


def test_vocab_size(tok):
    assert tok.vocab_size == 32_000


def test_special_token_ids_are_low(tok):
    assert tok.eof_id < 10
    assert tok.pad_id < 10
    assert tok.unk_id < 10


def test_special_token_ids_are_distinct(tok):
    assert len({tok.eof_id, tok.pad_id, tok.unk_id}) == 3


def test_len(tok):
    assert len(tok) == tok.vocab_size


def test_repr(tok):
    r = repr(tok)
    assert "SLMTokenizer" in r
    assert "vocab_size" in r


@pytest.mark.parametrize(
    "text",
    [
        "def __init__(self, x: int) -> None:",
        "import numpy as np\nimport torch",
        "    for i in range(len(self.layers)):",
        "self.attention = MultiHeadAttention(d_model, num_heads)",
        "x = {'key': [1, 2, 3]}",
        "# this is a comment\ndef foo():\n    pass",
    ],
)
def test_roundtrip(tok, text):
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text, f"Round-trip failed for: {repr(text)}"


def test_encode_returns_list_of_ints(tok):
    ids = tok.encode("def foo():")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_encode_nonempty(tok):
    ids = tok.encode("x = 1")
    assert len(ids) > 0


def test_add_eof_appends_eof_token(tok):
    ids = tok.encode("x = 1", add_eof=True)
    assert ids[-1] == tok.eof_id


def test_add_eof_false_does_not_append(tok):
    ids = tok.encode("x = 1", add_eof=False)
    assert ids[-1] != tok.eof_id


def test_encode_batch(tok):
    texts = ["def foo():", "import os", "x = 1"]
    ids_list = tok.encode_batch(texts)
    assert len(ids_list) == 3
    assert all(isinstance(ids, list) for ids in ids_list)


def test_encode_batch_matches_single(tok):
    texts = ["def foo():", "import os"]
    batch = tok.encode_batch(texts)
    singles = [tok.encode(t) for t in texts]
    assert batch == singles


def test_decode_batch(tok):
    texts = ["def foo():", "import os", "x = 1"]
    ids_list = tok.encode_batch(texts)
    decoded = tok.decode_batch(ids_list)
    assert decoded == texts


def test_skip_special_tokens_true(tok):
    ids = tok.encode("x = 1", add_eof=True)
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert "<|endoffile|>" not in decoded


def test_skip_special_tokens_false(tok):
    ids = tok.encode("x = 1", add_eof=True)
    decoded = tok.decode(ids, skip_special_tokens=False)
    assert "<|endoffile|>" in decoded


def test_token_to_id_special(tok):
    assert tok.token_to_id("<|endoffile|>") == tok.eof_id
    assert tok.token_to_id("<|pad|>") == tok.pad_id
    assert tok.token_to_id("<|unk|>") == tok.unk_id


def test_id_to_token_roundtrip(tok):
    token = tok.id_to_token(tok.eof_id)
    assert token == "<|endoffile|>"


def test_missing_tokenizer_raises():
    with pytest.raises(FileNotFoundError):
        SLMTokenizer("nonexistent/path")
