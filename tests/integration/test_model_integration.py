"""
tests/integration/test_model_integration.py

Integration tests for the full model stack. Unlike unit tests (which use a
tiny synthetic 2-layer config), these tests:

  - Load the real configs/model.yaml
  - Load the real trained tokenizer from tokenizer/trained/
  - Construct a full-scale TransformerModel (24 layers, d_model=512, 85M params)
  - Exercise the complete prefill → decode pipeline end-to-end
  - Verify cross-component contracts that unit tests cannot catch in isolation

Prerequisites:
    tokenizer/trained/tokenizer.json must exist.
    Run tokenizer/train.py first if it doesn't.

Run with:
    pytest tests/integration/test_model_integration.py -v
"""

from __future__ import annotations

import time

import pytest
import torch

from model.config import ModelConfig
from model.transformer import TransformerModel
from tokenizer.tokenizer import SLMTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_YAML = "configs/model.yaml"
TOKENIZER_DIR = "tokenizer/trained"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg() -> ModelConfig:
    return ModelConfig.from_yaml(MODEL_YAML)


@pytest.fixture(scope="module")
def tok() -> SLMTokenizer:
    if not __import__("pathlib").Path(TOKENIZER_DIR).joinpath("tokenizer.json").exists():
        pytest.skip("Trained tokenizer not found. Run tokenizer/train.py first.")
    return SLMTokenizer(TOKENIZER_DIR)


@pytest.fixture(scope="module")
def model(cfg) -> TransformerModel:
    """Full 70M model on CPU, weight-initialized, eval mode."""
    torch.manual_seed(0)
    m = TransformerModel(cfg)
    m.eval()
    return m


# ===========================================================================
# 1. Config ↔ Tokenizer contract
# ===========================================================================


class TestConfigTokenizerContract:
    def test_vocab_size_matches(self, cfg, tok):
        """cfg.vocab_size must equal the tokenizer's actual vocabulary size."""
        assert cfg.vocab_size == tok.vocab_size, (
            f"cfg.vocab_size={cfg.vocab_size} but tokenizer has {tok.vocab_size} tokens. "
            "Embedding matrix will be mis-sized."
        )

    def test_pad_token_exists_in_tokenizer(self, cfg, tok):
        pad_id = tok.token_to_id(cfg.pad_token)
        assert pad_id is not None, f"cfg.pad_token '{cfg.pad_token}' not found in tokenizer vocab"

    def test_eof_token_exists_in_tokenizer(self, cfg, tok):
        eof_id = tok.token_to_id(cfg.eof_token)
        assert eof_id is not None, f"cfg.eof_token '{cfg.eof_token}' not found in tokenizer vocab"

    def test_pad_token_id_matches_tokenizer_pad_id(self, cfg, tok):
        assert tok.token_to_id(cfg.pad_token) == tok.pad_id

    def test_eof_token_id_matches_tokenizer_eof_id(self, cfg, tok):
        assert tok.token_to_id(cfg.eof_token) == tok.eof_id

    def test_all_special_token_ids_are_valid_embedding_indices(self, cfg, tok):
        for token_id in [tok.pad_id, tok.eof_id, tok.unk_id]:
            assert (
                0 <= token_id < cfg.vocab_size
            ), f"Special token id {token_id} is out of range [0, {cfg.vocab_size})"


# ===========================================================================
# 2. Config ↔ Model construction
# ===========================================================================


class TestConfigModelContract:
    def test_model_constructs_from_real_yaml(self, cfg):
        """Full 24-layer model must construct without error from real config."""
        torch.manual_seed(1)
        m = TransformerModel(cfg)
        assert m is not None

    def test_param_count_is_near_85m(self, model):
        n = model.count_parameters()
        # Accept anything within 10% of 85M — config may land slightly off.
        assert 77_000_000 <= n <= 94_000_000, (
            f"Parameter count {n:,} is far from the 85M target. "
            "Check d_model, n_layer, d_ffn, vocab_size."
        )

    def test_embedding_shape(self, cfg, model):
        assert model.token_emb.weight.shape == (cfg.vocab_size, cfg.d_model)

    def test_n_blocks_matches_config(self, cfg, model):
        assert len(model.blocks) == cfg.n_layer

    def test_rope_max_seq_len_matches_config(self, cfg, model):
        assert model.rope.max_seq_len == cfg.context_length

    def test_final_norm_dim_matches_config(self, cfg, model):
        assert model.final_norm.weight.shape == (cfg.d_model,)


# ===========================================================================
# 3. Tokenizer → Model pipeline (encode then forward)
# ===========================================================================


class TestTokenizerToModelPipeline:
    PROMPT = "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n"

    def _encode_to_tensor(self, tok, text, max_len=None):
        ids = tok.encode(text)
        if max_len:
            ids = ids[:max_len]
        return torch.tensor([ids], dtype=torch.long)

    def test_encoded_ids_are_valid_model_inputs(self, cfg, tok, model):
        """Token IDs from the tokenizer must all be within the embedding range."""
        input_ids = self._encode_to_tensor(tok, self.PROMPT)
        assert input_ids.min() >= 0
        assert input_ids.max() < cfg.vocab_size

    def test_model_forward_on_real_tokens(self, cfg, tok, model):
        """Model must produce logits without error on real tokenized input."""
        input_ids = self._encode_to_tensor(tok, self.PROMPT)
        with torch.no_grad():
            logits, caches = model(input_ids)
        seq_len = input_ids.shape[1]
        assert logits.shape == (1, seq_len, cfg.vocab_size)

    def test_logits_are_finite(self, cfg, tok, model):
        input_ids = self._encode_to_tensor(tok, self.PROMPT)
        with torch.no_grad():
            logits, _ = model(input_ids)
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf on real tokenized input"

    def test_logits_vary_across_positions(self, tok, model):
        """Different token positions must produce different logit distributions."""
        input_ids = self._encode_to_tensor(tok, self.PROMPT)
        with torch.no_grad():
            logits, _ = model(input_ids)
        # First and last position logits should differ.
        assert not torch.allclose(
            logits[0, 0], logits[0, -1], atol=1e-3
        ), "Logits are identical across all positions — model may not be functioning correctly"

    def test_greedy_next_token_is_a_valid_id(self, cfg, tok, model):
        """The argmax next token must be a valid vocabulary index."""
        input_ids = self._encode_to_tensor(tok, self.PROMPT)
        with torch.no_grad():
            logits, _ = model(input_ids)
        next_token_id = logits[0, -1].argmax().item()
        assert 0 <= next_token_id < cfg.vocab_size

    def test_greedy_next_token_is_decodeable(self, tok, model, cfg):
        """The predicted next token must be decodeable by the tokenizer."""
        input_ids = self._encode_to_tensor(tok, self.PROMPT)
        with torch.no_grad():
            logits, _ = model(input_ids)
        next_token_id = logits[0, -1].argmax().item()
        token_str = tok.id_to_token(next_token_id)
        assert (
            token_str is not None
        ), f"Predicted token id {next_token_id} has no string representation in the tokenizer"


# ===========================================================================
# 4. Full prefill → decode loop
# ===========================================================================


class TestPrefillDecodeLoop:
    PROMPT = "def quicksort(arr):\n"
    N_DECODE_STEPS = 10

    def _prefill(self, tok, model, text, max_len=32):
        ids = tok.encode(text)[:max_len]
        input_ids = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits, caches = model(input_ids)
        return input_ids, logits, caches

    def test_prefill_succeeds(self, tok, model):
        _, logits, caches = self._prefill(tok, model, self.PROMPT)
        assert logits is not None
        assert caches is not None

    def test_decode_step_succeeds(self, cfg, tok, model):
        input_ids, logits, caches = self._prefill(tok, model, self.PROMPT)
        next_id = logits[0, -1].argmax(keepdim=True).unsqueeze(0)
        with torch.no_grad():
            logits2, caches2 = model(next_id, kv_caches=caches)
        assert logits2.shape == (1, 1, cfg.vocab_size)

    def test_n_decode_steps_without_error(self, cfg, tok, model):
        """Run N greedy decode steps; model must not error at any step."""
        input_ids, logits, caches = self._prefill(tok, model, self.PROMPT)

        for step in range(self.N_DECODE_STEPS):
            next_id = logits[0, -1].argmax(keepdim=True).unsqueeze(0)
            with torch.no_grad():
                logits, caches = model(next_id, kv_caches=caches)
            assert logits.shape == (1, 1, cfg.vocab_size), f"Bad shape at decode step {step}"
            assert torch.isfinite(logits).all(), f"Non-finite logits at decode step {step}"

    def test_kv_cache_grows_correctly_over_decode(self, tok, model):
        """After each decode step, KV cache seq_len must increment by 1."""
        input_ids, logits, caches = self._prefill(tok, model, self.PROMPT)
        prefill_len = input_ids.shape[1]

        for step in range(1, self.N_DECODE_STEPS + 1):
            next_id = logits[0, -1].argmax(keepdim=True).unsqueeze(0)
            with torch.no_grad():
                logits, caches = model(next_id, kv_caches=caches)
            expected_len = prefill_len + step
            actual_len = caches[0].seq_len
            assert (
                actual_len == expected_len
            ), f"Step {step}: expected cache seq_len={expected_len}, got {actual_len}"

    def test_all_layer_caches_have_same_seq_len(self, cfg, tok, model):
        """All layer KV caches must be at the same position after decode steps."""
        input_ids, logits, caches = self._prefill(tok, model, self.PROMPT)
        for _ in range(5):
            next_id = logits[0, -1].argmax(keepdim=True).unsqueeze(0)
            with torch.no_grad():
                logits, caches = model(next_id, kv_caches=caches)

        seq_lens = [c.seq_len for c in caches]
        assert len(set(seq_lens)) == 1, f"Layer KV caches have diverged seq_lens: {seq_lens}"

    def test_kv_cache_dtype_remains_bfloat16_throughout_decode(self, tok, model):
        """KV cache must stay bfloat16 across all decode steps (regression for RoPE upcast bug)."""
        input_ids, logits, caches = self._prefill(tok, model, self.PROMPT)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            ids = torch.tensor([tok.encode(self.PROMPT)[:16]], dtype=torch.long)
            logits, caches = model(ids)
            for _ in range(5):
                next_id = logits[0, -1].argmax(keepdim=True).unsqueeze(0)
                logits, caches = model(next_id, kv_caches=caches)

        for i, cache in enumerate(caches):
            assert (
                cache.k.dtype == torch.bfloat16
            ), f"Step 5, layer {i}: KV cache k is {cache.k.dtype}, expected bfloat16"
            assert (
                cache.v.dtype == torch.bfloat16
            ), f"Step 5, layer {i}: KV cache v is {cache.v.dtype}, expected bfloat16"

    def test_decode_output_differs_from_prefill_output(self, tok, model, cfg):
        """
        Logits from a decode step must not be identical to the prefill's last
        position — the cache-aware path must actually differ from a fresh prefill.
        """
        ids = torch.tensor([tok.encode(self.PROMPT)[:16]], dtype=torch.long)
        with torch.no_grad():
            logits_prefill, caches = model(ids)

        next_id = logits_prefill[0, -1].argmax(keepdim=True).unsqueeze(0)
        with torch.no_grad():
            logits_decode, _ = model(next_id, kv_caches=caches)

        # The decode logits are for the *new* token — they must differ from
        # the last prefill position's logits.
        assert not torch.allclose(logits_prefill[0, -1], logits_decode[0, 0], atol=1e-3), (
            "Decode step logits are identical to prefill last-position logits — "
            "KV cache may not be advancing position correctly"
        )

    def test_greedy_generation_produces_decodeable_sequence(self, cfg, tok, model):
        """All generated token IDs must be decodeable by the tokenizer."""
        ids = torch.tensor([tok.encode(self.PROMPT)[:16]], dtype=torch.long)
        generated = []

        with torch.no_grad():
            logits, caches = model(ids)
            for _ in range(self.N_DECODE_STEPS):
                next_id = logits[0, -1].argmax().item()
                generated.append(next_id)
                next_tensor = torch.tensor([[next_id]])
                logits, caches = model(next_tensor, kv_caches=caches)

        # Every generated id must map to a token string.
        for token_id in generated:
            assert 0 <= token_id < cfg.vocab_size
            assert (
                tok.id_to_token(token_id) is not None
            ), f"Generated token id {token_id} is not in the tokenizer vocabulary"

        # The whole sequence must decode without error.
        decoded = tok.decode(generated, skip_special_tokens=False)
        assert isinstance(decoded, str)


# ===========================================================================
# 5. Autocast + model forward compatibility
# ===========================================================================


class TestAutocastCompatibility:
    def test_forward_under_autocast_does_not_error(self, tok, model, cfg):
        ids = torch.tensor([tok.encode("import os\n")[:8]], dtype=torch.long)
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits, caches = model(ids)
        assert torch.isfinite(logits).all()

    def test_prefill_and_decode_under_autocast(self, tok, model, cfg):
        ids = torch.tensor([tok.encode("class Node:\n")[:12]], dtype=torch.long)
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits, caches = model(ids)
            next_id = logits[0, -1].argmax(keepdim=True).unsqueeze(0)
            logits2, _ = model(next_id, kv_caches=caches)
        assert logits2.shape == (1, 1, cfg.vocab_size)
        assert torch.isfinite(logits2).all()

    def test_logits_shape_consistent_with_and_without_autocast(self, tok, model, cfg):
        ids = torch.tensor([tok.encode("x = 1\n")[:6]], dtype=torch.long)
        with torch.no_grad():
            logits_fp32, _ = model(ids)
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits_bf16, _ = model(ids)
        assert logits_fp32.shape == logits_bf16.shape


# ===========================================================================
# 6. Context length boundary
# ===========================================================================


class TestContextLengthBoundary:
    def test_full_context_length_prefill(self, cfg, model):
        """Model must handle a prefill of exactly context_length tokens."""
        ids = torch.randint(0, cfg.vocab_size, (1, cfg.context_length))
        with torch.no_grad():
            logits, caches = model(ids)
        assert logits.shape == (1, cfg.context_length, cfg.vocab_size)
        assert torch.isfinite(logits).all()

    def test_rope_does_not_error_at_max_seq_len(self, cfg, model):
        """RoPE cache must cover exactly context_length positions without raising."""
        ids = torch.randint(0, cfg.vocab_size, (1, cfg.context_length))
        with torch.no_grad():
            model(ids)  # should not raise

    def test_exceeding_context_length_raises(self, cfg, model):
        """Requesting more positions than context_length must raise ValueError."""
        ids = torch.randint(0, cfg.vocab_size, (1, cfg.context_length + 1))
        with pytest.raises(ValueError, match="exceed"), torch.no_grad():
            model(ids)


# ===========================================================================
# 7. Determinism
# ===========================================================================


class TestDeterminism:
    def test_same_input_same_logits(self, tok, model):
        """Identical inputs must produce identical outputs (no stochastic ops in eval)."""
        ids = torch.tensor([tok.encode("def foo():\n    pass\n")[:12]], dtype=torch.long)
        with torch.no_grad():
            logits1, _ = model(ids)
            logits2, _ = model(ids)
        assert torch.equal(logits1, logits2), "Non-deterministic output for identical inputs"

    def test_same_input_same_kv_cache(self, tok, model, cfg):
        """KV cache produced from identical inputs must be identical."""
        ids = torch.tensor([tok.encode("x = 1\n")[:6]], dtype=torch.long)
        with torch.no_grad():
            _, caches1 = model(ids)
            _, caches2 = model(ids)
        for i in range(cfg.n_layer):
            assert torch.equal(caches1[i].k, caches2[i].k), f"Layer {i} k cache differs"
            assert torch.equal(caches1[i].v, caches2[i].v), f"Layer {i} v cache differs"

    def test_greedy_generation_is_deterministic(self, tok, model, cfg):
        """Two identical greedy generation runs must produce the same sequence."""

        def generate(n_steps=5):
            ids = torch.tensor([tok.encode("return x +")[:8]], dtype=torch.long)
            generated = []
            with torch.no_grad():
                logits, caches = model(ids)
                for _ in range(n_steps):
                    next_id = logits[0, -1].argmax().item()
                    generated.append(next_id)
                    logits, caches = model(torch.tensor([[next_id]]), kv_caches=caches)
            return generated

        assert generate() == generate(), "Greedy generation is non-deterministic"


# ===========================================================================
# 8. Performance smoke test (soft — warns, does not fail)
# ===========================================================================


class TestPerformance:
    def test_prefill_completes_in_reasonable_time(self, tok, model):
        """
        A 32-token prefill on CPU should complete in under 60 seconds.
        This is a very loose bound — just catches catastrophic regressions
        like accidentally running in double precision or with autograd enabled.
        """
        ids = torch.tensor([tok.encode("def foo():\n    return 1\n")[:32]], dtype=torch.long)
        start = time.perf_counter()
        with torch.no_grad():
            model(ids)
        elapsed = time.perf_counter() - start
        assert elapsed < 60.0, (
            f"32-token CPU prefill took {elapsed:.1f}s — suspiciously slow. "
            "Check for double precision or accidental autograd."
        )

    def test_decode_step_faster_than_full_prefill(self, tok, model):
        """
        A single-token decode step should be faster than a full 32-token prefill,
        since the KV cache skips recomputing previous context.
        """
        prompt_ids = torch.tensor([tok.encode("x = [i for i in range(10)]")[:16]], dtype=torch.long)

        # Prefill timing
        with torch.no_grad():
            t0 = time.perf_counter()
            logits, caches = model(prompt_ids)
            prefill_time = time.perf_counter() - t0

        # Decode timing (average over 3 steps)
        decode_times = []
        for _ in range(3):
            next_id = logits[0, -1].argmax(keepdim=True).unsqueeze(0)
            with torch.no_grad():
                t0 = time.perf_counter()
                logits, caches = model(next_id, kv_caches=caches)
                decode_times.append(time.perf_counter() - t0)

        avg_decode = sum(decode_times) / len(decode_times)
        assert avg_decode < prefill_time, (
            f"Decode step ({avg_decode:.3f}s) is not faster than prefill ({prefill_time:.3f}s). "
            "KV cache may not be working correctly."
        )
