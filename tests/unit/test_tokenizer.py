"""
tests/unit/test_tokenizer.py

Unit tests for SLMTokenizer wrapper and tokenizer artifacts.
Requires a trained tokenizer at tokenizer/trained/.
Run tokenizer/train.py first, then: pytest tests/unit/test_tokenizer.py
"""

import json
from pathlib import Path

import pytest

from tokenizer.tokenizer import SLMTokenizer

TOKENIZER_DIR = Path("tokenizer/trained")
REQUIRED_ARTIFACTS = ["tokenizer.json", "tokenizer_meta.json", "vocab.txt", "training_log.json"]
REQUIRED_LOG_KEYS = [
    "trained_at",
    "duration_seconds",
    "vocab_size",
    "dataset",
    "max_files",
    "files_collected",
    "files_skipped_short",
    "total_chars",
    "avg_file_length",
    "special_tokens",
    "tokenizers_version",
]


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


def test_empty_string(tok):
    ids = tok.encode("")
    assert isinstance(ids, list)
    assert len(ids) == 0


def test_whitespace_only(tok):
    ids = tok.encode("     ")
    assert isinstance(ids, list)
    decoded = tok.decode(ids)
    assert decoded == "     "


def test_single_character(tok):
    ids = tok.encode("x")
    assert len(ids) >= 1
    assert tok.decode(ids) == "x"


def test_newline_only(tok):
    ids = tok.encode("\n")
    assert isinstance(ids, list)
    assert tok.decode(ids) == "\n"


def test_only_comments(tok):
    text = "# this is a comment\n# another comment\n"
    ids = tok.encode(text)
    assert len(ids) > 0
    assert tok.decode(ids) == text


def test_mixed_indentation(tok):
    text = "def foo():\n    x = 1\n\tx = 2\n"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_very_long_file(tok):
    text = "x = 1\n" * 10_000
    ids = tok.encode(text)
    assert len(ids) > 0
    assert tok.decode(ids) == text


def test_unicode_accented(tok):
    text = "# café résumé naïve\nx = 1\n"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_unicode_cjk(tok):
    text = "# 中文注释\nx = 1\n"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_unicode_emoji(tok):
    text = "# 🐍 python\nx = 1\n"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_multiline_string(tok):
    text = '"""\\nThis is a\\ndocstring\\n"""\\n'
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_deeply_nested_code(tok):
    text = "if a:\n    if b:\n        if c:\n            if d:\n                x = 1\n"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_all_ids_in_valid_range(tok):
    text = "def foo(x: int) -> None:\n    return x * 2\n"
    ids = tok.encode(text)
    assert all(0 <= i < tok.vocab_size for i in ids), "Token ID out of valid range"


def test_special_token_ids_in_range(tok):
    assert 0 <= tok.eof_id < tok.vocab_size
    assert 0 <= tok.pad_id < tok.vocab_size
    assert 0 <= tok.unk_id < tok.vocab_size


@pytest.mark.parametrize(
    "text",
    [
        "def foo(): pass",
        "import os\nimport sys\n",
        "class MyModel(nn.Module):\n    pass\n",
    ],
)
def test_encode_is_deterministic(tok, text):
    ids1 = tok.encode(text)
    ids2 = tok.encode(text)
    assert ids1 == ids2, f"Non-deterministic encoding for: {repr(text)}"


def test_reload_produces_identical_ids(tok):
    """Tokenizer reloaded from disk must produce identical IDs."""
    text = "def __init__(self, hidden_dim: int) -> None:\n    super().__init__()\n"
    ids_original = tok.encode(text)
    reloaded = SLMTokenizer(TOKENIZER_DIR)
    ids_reloaded = reloaded.encode(text)
    assert ids_original == ids_reloaded, "Reloaded tokenizer produces different IDs"


FERTILITY_SAMPLE = '''
def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


class Stack:
    def __init__(self) -> None:
        self._items: list = []

    def push(self, item) -> None:
        self._items.append(item)

    def pop(self):
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()
'''


def test_fertility_above_threshold(tok):
    """Chars per token should be > 2.0 for a Python-trained tokenizer."""
    ids = tok.encode(FERTILITY_SAMPLE)
    ratio = len(FERTILITY_SAMPLE) / len(ids)
    assert ratio > 2.0, f"Fertility too low: {ratio:.2f} chars/token (expected > 2.0)"


def test_four_space_indent_is_single_token(tok):
    """4-space indent must be a single token — critical for Python."""
    ids = tok.encode("    ")
    assert len(ids) == 1, f"4-space indent tokenized to {len(ids)} tokens, expected 1"


@pytest.mark.parametrize(
    "keyword",
    [
        "None",
        "True",
        "False",
        "return",
        "import",
        "class",
        "def",
    ],
)
def test_python_keywords_few_tokens(tok, keyword):
    ids = tok.encode(keyword)
    assert len(ids) <= 3, f"'{keyword}' tokenized to {len(ids)} tokens, expected ≤ 3"


def test_all_artifacts_exist():
    for artifact in REQUIRED_ARTIFACTS:
        path = TOKENIZER_DIR / artifact
        assert path.exists(), f"Missing artifact: {artifact}"


def test_training_log_has_required_keys():
    log_path = TOKENIZER_DIR / "training_log.json"
    if not log_path.exists():
        pytest.skip("training_log.json not found.")
    with open(log_path) as f:
        log = json.load(f)
    for key in REQUIRED_LOG_KEYS:
        assert key in log, f"Missing key in training_log.json: '{key}'"


def test_training_log_vocab_size_matches(tok):
    log_path = TOKENIZER_DIR / "training_log.json"
    if not log_path.exists():
        pytest.skip("training_log.json not found.")
    with open(log_path) as f:
        log = json.load(f)
    assert log["vocab_size"] == tok.vocab_size


def test_training_log_files_collected_positive():
    log_path = TOKENIZER_DIR / "training_log.json"
    if not log_path.exists():
        pytest.skip("training_log.json not found.")
    with open(log_path) as f:
        log = json.load(f)
    assert log["files_collected"] > 0
    assert log["total_chars"] > 0
    assert log["duration_seconds"] > 0


def test_meta_vocab_size_matches(tok):
    meta_path = TOKENIZER_DIR / "tokenizer_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["vocab_size"] == tok.vocab_size


def test_meta_special_tokens_match(tok):
    meta_path = TOKENIZER_DIR / "tokenizer_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    special = meta["special_tokens"]
    assert special["<|endoffile|>"] == tok.eof_id
    assert special["<|pad|>"] == tok.pad_id
    assert special["<|unk|>"] == tok.unk_id


def test_vocab_txt_line_count(tok):
    vocab_path = TOKENIZER_DIR / "vocab.txt"
    lines = vocab_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == tok.vocab_size


REGRESSION_FILE = Path("tests/unit/tokenizer_regression.json")


def test_regression_set_exists():
    """Regression file must exist. Generate with scripts/generate_regression.py"""
    assert (
        REGRESSION_FILE.exists()
    ), "Regression file missing. Run: python scripts/generate_regression.py"


def test_regression_ids_stable(tok):
    if not REGRESSION_FILE.exists():
        pytest.skip("Regression file not found.")
    with open(REGRESSION_FILE) as f:
        cases = json.load(f)
    for case in cases:
        ids = tok.encode(case["text"])
        assert ids == case["ids"], (
            f"Regression failure for: {repr(case['text'])}\n"
            f"  expected: {case['ids'][:10]}...\n"
            f"  got:      {ids[:10]}..."
        )
