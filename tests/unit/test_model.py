"""
tests/unit/test_model.py

Comprehensive unit tests for every module in model/.

Coverage:
  - ModelConfig: loading, validation, derived properties
  - RMSNorm: output shape, dtype preservation, normalization math, weight scale
  - RotaryEmbedding / apply_rotary_pos_emb: shape, dtype, value, cache consistency
  - KVCache: seq_len, shape tracking
  - GQAAttention: shape, KV cache round-trip, dtype correctness of cache
  - SwiGLUFFN: shape, dtype, dropout behaviour
  - TransformerBlock: shape, residual stream, cache passthrough
  - TransformerModel: construction, param count, weight tying, forward shape,
      prefill + decode dtype, kv_cache length guard, SCALE_RESIDUAL_INIT scaling

All tests use a tiny synthetic config (2 layers, d_model=64) so they run
in milliseconds on CPU without a GPU or trained tokenizer.

Run with:
    pytest tests/unit/test_model.py -v
"""

from __future__ import annotations

import pytest
import torch

from model.attention import GQAAttention, KVCache
from model.block import TransformerBlock
from model.config import ModelConfig
from model.ffn import SwiGLUFFN
from model.rmsnorm import RMSNorm
from model.rope import RotaryEmbedding, apply_rotary_pos_emb
from model.transformer import TransformerModel

TINY_CFG_DICT = {
    "n_layer": 2,
    "d_model": 64,
    "n_heads_q": 4,
    "n_heads_kv": 2,
    "head_dim": 16,  # 4 * 16 == 64 ✓
    "d_ffn": 128,
    "vocab_size": 256,
    "context_length": 32,
    "dropout": 0.0,
    "dtype": torch.bfloat16,
    "tokenizer_path": "tokenizer/trained",  # doesn't need to exist for model tests
    "pad_token": "<|pad|>",
    "eof_token": "<|endoffile|>",
}


@pytest.fixture(scope="module")
def cfg() -> ModelConfig:
    return ModelConfig(**TINY_CFG_DICT)


@pytest.fixture(scope="module")
def model(cfg) -> TransformerModel:
    torch.manual_seed(0)
    return TransformerModel(cfg)


class TestModelConfig:
    def test_construct_from_dict(self, cfg):
        assert cfg.n_layer == 2
        assert cfg.d_model == 64

    def test_gqa_group_size(self, cfg):
        assert cfg.gqa_group_size == cfg.n_heads_q // cfg.n_heads_kv  # 2

    def test_kv_dim(self, cfg):
        assert cfg.kv_dim == cfg.n_heads_kv * cfg.head_dim  # 32

    def test_n_heads_q_times_head_dim_equals_d_model(self, cfg):
        assert cfg.n_heads_q * cfg.head_dim == cfg.d_model

    def test_validate_rejects_bad_head_dim(self):
        bad = dict(TINY_CFG_DICT)
        bad["head_dim"] = 15  # 4 * 15 != 64
        with pytest.raises(ValueError, match="n_heads_q"):
            ModelConfig(**bad)._validate()

    def test_validate_rejects_non_divisible_gqa(self):
        bad = dict(TINY_CFG_DICT)
        bad["n_heads_kv"] = 3  # 4 % 3 != 0
        with pytest.raises(ValueError, match="multiple"):
            ModelConfig(**bad)._validate()

    def test_validate_rejects_bad_dropout(self):
        bad = dict(TINY_CFG_DICT)
        bad["dropout"] = 1.5
        with pytest.raises(ValueError, match="dropout"):
            ModelConfig(**bad)._validate()

    def test_from_yaml_loads_correctly(self, tmp_path):
        yaml_text = (
            "n_layer: 2\n"
            "d_model: 64\n"
            "n_heads_q: 4\n"
            "n_heads_kv: 2\n"
            "head_dim: 16\n"
            "d_ffn: 128\n"
            "vocab_size: 256\n"
            "context_length: 32\n"
            "dropout: 0.0\n"
            "dtype: bfloat16\n"
            "tokenizer_path: tokenizer/trained\n"
            'pad_token: "<|pad|>"\n'
            'eof_token: "<|endoffile|>"\n'
        )
        p = tmp_path / "model.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        loaded = ModelConfig.from_yaml(p)
        assert loaded.n_layer == 2
        assert loaded.dtype == torch.bfloat16

    def test_from_yaml_rejects_unknown_dtype(self, tmp_path):
        yaml_text = (
            "n_layer: 2\nd_model: 64\nn_heads_q: 4\nn_heads_kv: 2\n"
            "head_dim: 16\nd_ffn: 128\nvocab_size: 256\ncontext_length: 32\n"
            "dropout: 0.0\ndtype: float64\n"
            "tokenizer_path: x\npad_token: p\neof_token: e\n"
        )
        p = tmp_path / "bad.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(ValueError, match="unknown dtype"):
            ModelConfig.from_yaml(p)


class TestRMSNorm:
    @pytest.fixture
    def norm(self):
        return RMSNorm(dim=64)

    def test_output_shape_preserved(self, norm):
        x = torch.randn(2, 10, 64)
        assert norm(x).shape == x.shape

    def test_output_dtype_float32_in(self, norm):
        x = torch.randn(2, 10, 64)
        assert norm(x).dtype == torch.float32

    def test_output_dtype_bfloat16_preserved(self, norm):
        x = torch.randn(2, 10, 64).bfloat16()
        assert norm(x).dtype == torch.bfloat16

    def test_output_dtype_float16_preserved(self, norm):
        x = torch.randn(2, 10, 64).half()
        assert norm(x).dtype == torch.float16

    def test_rms_of_output_is_near_one(self, norm):
        """With default weight=1, each output vector's RMS should be ~1."""
        x = torch.randn(4, 16, 64)
        y = norm(x)
        rms = y.float().pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_weight_scale_is_respected(self):
        """Setting weight to 2 should double the output magnitude."""
        norm = RMSNorm(dim=64)
        with torch.no_grad():
            norm.weight.fill_(2.0)
        x = torch.randn(2, 8, 64)
        y = norm(x)
        rms = y.float().pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.full_like(rms, 2.0), atol=1e-4)

    def test_weight_is_learnable_parameter(self, norm):
        assert isinstance(norm.weight, torch.nn.Parameter)
        assert norm.weight.requires_grad

    def test_initialised_to_ones(self, norm):
        assert torch.all(norm.weight == 1.0)

    def test_no_nan_in_output(self, norm):
        x = torch.randn(2, 10, 64)
        assert not torch.isnan(norm(x)).any()

    def test_zero_input_does_not_produce_nan(self, norm):
        """rsqrt(0 + eps) should be finite, not nan/inf."""
        x = torch.zeros(2, 4, 64)
        y = norm(x)
        assert torch.isfinite(y).all()

    def test_3d_and_2d_input(self, norm):
        x3 = torch.randn(2, 5, 64)
        x2 = torch.randn(5, 64)
        assert norm(x3).shape == (2, 5, 64)
        assert norm(x2).shape == (5, 64)

    def test_gradient_flows(self, norm):
        x = torch.randn(2, 4, 64, requires_grad=True)
        y = norm(x)
        y.sum().backward()
        assert x.grad is not None
        assert norm.weight.grad is not None


class TestRotaryEmbedding:
    @pytest.fixture
    def rope(self):
        return RotaryEmbedding(head_dim=16, max_seq_len=32)

    def test_output_shapes(self, rope):
        cos, sin = rope(seq_len=8, device=torch.device("cpu"))
        assert cos.shape == (8, 16)
        assert sin.shape == (8, 16)

    def test_output_dtype_is_float32(self, rope):
        """Cache tables are kept in fp32 for precision."""
        cos, sin = rope(seq_len=8, device=torch.device("cpu"))
        assert cos.dtype == torch.float32
        assert sin.dtype == torch.float32

    def test_start_pos_offset(self, rope):
        """Slices at different start_pos must differ."""
        cos0, _ = rope(seq_len=4, device=torch.device("cpu"), start_pos=0)
        cos4, _ = rope(seq_len=4, device=torch.device("cpu"), start_pos=4)
        assert not torch.allclose(cos0, cos4)

    def test_start_pos_continuity(self, rope):
        """Concatenation of two slices must equal one contiguous slice."""
        cos_full, sin_full = rope(seq_len=8, device=torch.device("cpu"), start_pos=0)
        cos_a, sin_a = rope(seq_len=4, device=torch.device("cpu"), start_pos=0)
        cos_b, sin_b = rope(seq_len=4, device=torch.device("cpu"), start_pos=4)
        assert torch.allclose(torch.cat([cos_a, cos_b], dim=0), cos_full)
        assert torch.allclose(torch.cat([sin_a, sin_b], dim=0), sin_full)

    def test_exceeding_max_seq_len_raises(self, rope):
        with pytest.raises(ValueError, match="exceed"):
            rope(seq_len=33, device=torch.device("cpu"))

    def test_rejects_odd_head_dim(self):
        with pytest.raises(ValueError, match="even"):
            RotaryEmbedding(head_dim=15, max_seq_len=32)

    def test_no_learned_parameters(self, rope):
        assert sum(p.numel() for p in rope.parameters()) == 0

    def test_cos_sin_are_unit_circle(self, rope):
        """For each position and pair, cos^2 + sin^2 must equal 1."""
        cos, sin = rope(seq_len=8, device=torch.device("cpu"))
        identity = cos**2 + sin**2
        assert torch.allclose(identity, torch.ones_like(identity), atol=1e-5)


class TestApplyRotaryPosEmb:
    @pytest.fixture
    def rope(self):
        return RotaryEmbedding(head_dim=16, max_seq_len=32)

    def _make_qk(self, b=2, n_heads=4, t=6, head_dim=16, dtype=torch.float32):
        torch.manual_seed(42)
        q = torch.randn(b, n_heads, t, head_dim, dtype=dtype)
        k = torch.randn(b, n_heads, t, head_dim, dtype=dtype)
        return q, k

    def test_output_shapes_unchanged(self, rope):
        q, k = self._make_qk()
        cos, sin = rope(seq_len=6, device=torch.device("cpu"))
        q_r, k_r = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_r.shape == q.shape
        assert k_r.shape == k.shape

    def test_output_dtype_matches_input_float32(self, rope):
        q, k = self._make_qk(dtype=torch.float32)
        cos, sin = rope(seq_len=6, device=torch.device("cpu"))
        q_r, k_r = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_r.dtype == torch.float32
        assert k_r.dtype == torch.float32

    def test_output_dtype_matches_input_bfloat16(self, rope):
        """Core regression: KV cache must not silently upcast to float32."""
        q, k = self._make_qk(dtype=torch.bfloat16)
        cos, sin = rope(seq_len=6, device=torch.device("cpu"))
        q_r, k_r = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_r.dtype == torch.bfloat16, (
            f"q_rotated is {q_r.dtype}, expected bfloat16 — "
            "RoPE dtype cast is broken, KV cache will use 2x memory"
        )
        assert k_r.dtype == torch.bfloat16

    def test_output_dtype_matches_input_float16(self, rope):
        q, k = self._make_qk(dtype=torch.float16)
        cos, sin = rope(seq_len=6, device=torch.device("cpu"))
        q_r, k_r = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_r.dtype == torch.float16
        assert k_r.dtype == torch.float16

    def test_rotation_is_not_identity(self, rope):
        """RoPE should actually change q/k values."""
        q, k = self._make_qk()
        cos, sin = rope(seq_len=6, device=torch.device("cpu"))
        q_r, k_r = apply_rotary_pos_emb(q, k, cos, sin)
        assert not torch.allclose(q, q_r)
        assert not torch.allclose(k, k_r)

    def test_rotation_preserves_norm(self, rope):
        """RoPE is a rotation, so ||q_rotated|| == ||q|| for each vector."""
        q, k = self._make_qk(dtype=torch.float32)
        cos, sin = rope(seq_len=6, device=torch.device("cpu"))
        q_r, k_r = apply_rotary_pos_emb(q, k, cos, sin)
        norms_q = q.norm(dim=-1)
        norms_q_r = q_r.norm(dim=-1)
        assert torch.allclose(norms_q, norms_q_r, atol=1e-5)

    def test_relative_position_property(self, rope):
        """
        q[pos=m] · k[pos=n] should depend only on (m-n).
        Verify: q[0]·k[0] ≈ q[1]·k[1] when q and k are identical vectors
        (both at relative offset 0).
        """
        head_dim = 16
        v = torch.ones(1, 1, 1, head_dim)
        cos0, sin0 = rope(seq_len=1, device=torch.device("cpu"), start_pos=0)
        cos1, sin1 = rope(seq_len=1, device=torch.device("cpu"), start_pos=1)
        q0, _ = apply_rotary_pos_emb(v, v, cos0, sin0)
        q1, _ = apply_rotary_pos_emb(v, v, cos1, sin1)
        k0, _ = apply_rotary_pos_emb(v, v, cos0, sin0)
        k1, _ = apply_rotary_pos_emb(v, v, cos1, sin1)
        dot00 = (q0 * k0).sum().item()
        dot11 = (q1 * k1).sum().item()
        assert (
            abs(dot00 - dot11) < 1e-4
        ), "RoPE dot product should be position-invariant for equal offset"


class TestKVCache:
    def _make_cache(self, b=2, n_kv=2, t=5, hd=16, dtype=torch.bfloat16):
        k = torch.randn(b, n_kv, t, hd, dtype=dtype)
        v = torch.randn(b, n_kv, t, hd, dtype=dtype)
        return KVCache(k=k, v=v)

    def test_seq_len(self):
        cache = self._make_cache(t=5)
        assert cache.seq_len == 5

    def test_seq_len_after_concat(self):
        cache = self._make_cache(t=5)
        extra_k = torch.randn(2, 2, 3, 16, dtype=torch.bfloat16)
        extra_v = torch.randn(2, 2, 3, 16, dtype=torch.bfloat16)
        new_k = torch.cat([cache.k, extra_k], dim=2)
        new_v = torch.cat([cache.v, extra_v], dim=2)
        new_cache = KVCache(k=new_k, v=new_v)
        assert new_cache.seq_len == 8

    def test_shape_fields(self):
        cache = self._make_cache(b=3, n_kv=4, t=7, hd=16)
        assert cache.k.shape == (3, 4, 7, 16)
        assert cache.v.shape == (3, 4, 7, 16)

    def test_dtype_preserved(self):
        cache = self._make_cache(dtype=torch.bfloat16)
        assert cache.k.dtype == torch.bfloat16
        assert cache.v.dtype == torch.bfloat16


class TestGQAAttention:
    @pytest.fixture
    def rope(self, cfg):
        return RotaryEmbedding(cfg.head_dim, cfg.context_length)

    @pytest.fixture
    def attn(self, cfg):
        torch.manual_seed(1)
        return GQAAttention(cfg)

    def _cos_sin(self, rope, seq_len, start_pos=0):
        return rope(seq_len=seq_len, device=torch.device("cpu"), start_pos=start_pos)

    def test_output_shape_no_cache(self, cfg, attn, rope):
        x = torch.randn(2, 6, cfg.d_model)
        cos, sin = self._cos_sin(rope, 6)
        out, cache = attn(x, cos, sin, kv_cache=None)
        assert out.shape == (2, 6, cfg.d_model)

    def test_returns_kv_cache(self, cfg, attn, rope):
        x = torch.randn(2, 6, cfg.d_model)
        cos, sin = self._cos_sin(rope, 6)
        _, cache = attn(x, cos, sin, kv_cache=None)
        assert isinstance(cache, KVCache)

    def test_kv_cache_shape_after_prefill(self, cfg, attn, rope):
        x = torch.randn(2, 6, cfg.d_model)
        cos, sin = self._cos_sin(rope, 6)
        _, cache = attn(x, cos, sin, kv_cache=None)
        assert cache.k.shape == (2, cfg.n_heads_kv, 6, cfg.head_dim)
        assert cache.v.shape == (2, cfg.n_heads_kv, 6, cfg.head_dim)

    def test_kv_cache_grows_on_decode(self, cfg, attn, rope):
        """seq_len in the cache should grow by 1 each decode step."""
        x = torch.randn(2, 4, cfg.d_model)
        cos, sin = self._cos_sin(rope, 4)
        _, cache = attn(x, cos, sin, kv_cache=None)
        assert cache.seq_len == 4

        x_next = torch.randn(2, 1, cfg.d_model)
        cos1, sin1 = self._cos_sin(rope, 1, start_pos=4)
        _, cache2 = attn(x_next, cos1, sin1, kv_cache=cache)
        assert cache2.seq_len == 5

    def test_kv_cache_dtype_is_not_float32_with_bfloat16_input(self, cfg, attn, rope):
        """
        Regression test for RoPE float32 upcast bug.
        k/v in the cache must stay bfloat16 when run under autocast (the
        normal training mode — weights stay fp32, matmuls run in bf16).
        """
        x = torch.randn(2, 4, cfg.d_model)
        cos, sin = self._cos_sin(rope, 4)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            _, cache = attn(x, cos, sin, kv_cache=None)
        assert cache.k.dtype == torch.bfloat16, (
            f"KV cache k is {cache.k.dtype}, expected bfloat16. "
            "RoPE dtype promotion bug is back."
        )
        assert cache.v.dtype == torch.bfloat16

    def test_output_dtype_matches_input_under_autocast(self, cfg, attn, rope):
        """Output dtype must be bfloat16 when running under autocast."""
        x = torch.randn(2, 4, cfg.d_model)
        cos, sin = self._cos_sin(rope, 4)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out, _ = attn(x, cos, sin, kv_cache=None)
        assert out.dtype == torch.bfloat16

    def test_no_nan_in_output(self, cfg, attn, rope):
        x = torch.randn(2, 6, cfg.d_model)
        cos, sin = self._cos_sin(rope, 6)
        out, _ = attn(x, cos, sin, kv_cache=None)
        assert not torch.isnan(out).any()

    def test_dropout_off_in_eval(self, cfg, rope):
        """Two identical forward passes in eval mode must be identical."""
        torch.manual_seed(5)
        attn = GQAAttention(ModelConfig(**{**TINY_CFG_DICT, "dropout": 0.5}))
        attn.eval()
        x = torch.randn(2, 4, cfg.d_model)
        cos, sin = self._cos_sin(rope, 4)
        out1, _ = attn(x, cos, sin)
        out2, _ = attn(x, cos, sin)
        assert torch.allclose(out1, out2)

    def test_o_proj_tagged_scale_residual_init(self, attn):
        assert getattr(attn.o_proj, "SCALE_RESIDUAL_INIT", False) is True

    def test_q_proj_not_tagged(self, attn):
        assert not getattr(attn.q_proj, "SCALE_RESIDUAL_INIT", False)

    def test_no_bias_on_projections(self, attn):
        for proj in [attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj]:
            assert proj.bias is None


class TestSwiGLUFFN:
    @pytest.fixture
    def ffn(self, cfg):
        torch.manual_seed(2)
        return SwiGLUFFN(cfg)

    def test_output_shape(self, cfg, ffn):
        x = torch.randn(2, 8, cfg.d_model)
        assert ffn(x).shape == (2, 8, cfg.d_model)

    def test_output_dtype_preserved_float32(self, ffn, cfg):
        x = torch.randn(2, 4, cfg.d_model)
        assert ffn(x).dtype == torch.float32

    def test_output_dtype_preserved_bfloat16(self, ffn, cfg):
        x = torch.randn(2, 4, cfg.d_model).bfloat16()
        ffn.bfloat16()
        assert ffn(x).dtype == torch.bfloat16

    def test_no_nan_in_output(self, ffn, cfg):
        x = torch.randn(2, 6, cfg.d_model)
        assert not torch.isnan(ffn(x)).any()

    def test_down_proj_tagged_scale_residual_init(self, ffn):
        assert getattr(ffn.down_proj, "SCALE_RESIDUAL_INIT", False) is True

    def test_gate_up_proj_not_tagged(self, ffn):
        assert not getattr(ffn.gate_proj, "SCALE_RESIDUAL_INIT", False)
        assert not getattr(ffn.up_proj, "SCALE_RESIDUAL_INIT", False)

    def test_no_bias_on_any_proj(self, ffn):
        for proj in [ffn.gate_proj, ffn.up_proj, ffn.down_proj]:
            assert proj.bias is None

    def test_dropout_off_in_eval(self, cfg):
        """Two forward passes in eval mode must give identical results."""
        torch.manual_seed(9)
        ffn = SwiGLUFFN(ModelConfig(**{**TINY_CFG_DICT, "dropout": 0.5}))
        ffn.eval()
        x = torch.randn(2, 4, cfg.d_model)
        assert torch.allclose(ffn(x), ffn(x))

    def test_gradient_flows(self, ffn, cfg):
        x = torch.randn(2, 4, cfg.d_model, requires_grad=True)
        y = ffn(x)
        y.sum().backward()
        assert x.grad is not None

    def test_swiglu_is_not_just_linear(self, ffn, cfg):
        """SwiGLU output must be non-linear: f(2x) != 2*f(x) for generic x."""
        x = torch.randn(1, 1, cfg.d_model)
        y1 = ffn(x)
        y2 = ffn(2 * x)
        assert not torch.allclose(y2, 2 * y1, atol=1e-3)


class TestTransformerBlock:
    @pytest.fixture
    def rope(self, cfg):
        return RotaryEmbedding(cfg.head_dim, cfg.context_length)

    @pytest.fixture
    def block(self, cfg):
        torch.manual_seed(3)
        return TransformerBlock(cfg)

    def _cos_sin(self, rope, t, start_pos=0):
        return rope(seq_len=t, device=torch.device("cpu"), start_pos=start_pos)

    def test_output_shape(self, cfg, block, rope):
        x = torch.randn(2, 5, cfg.d_model)
        cos, sin = self._cos_sin(rope, 5)
        out, _ = block(x, cos, sin)
        assert out.shape == (2, 5, cfg.d_model)

    def test_returns_kv_cache(self, cfg, block, rope):
        x = torch.randn(2, 5, cfg.d_model)
        cos, sin = self._cos_sin(rope, 5)
        _, cache = block(x, cos, sin)
        assert isinstance(cache, KVCache)

    def test_residual_connection_present(self, cfg, rope):
        """With zero-init attn and ffn weights output ≈ input (residual path)."""
        torch.manual_seed(7)
        block = TransformerBlock(cfg)
        # Zero out all sub-layer weights so only the residual stream survives.
        with torch.no_grad():
            for p in block.attn.parameters():
                p.zero_()
            for p in block.ffn.parameters():
                p.zero_()
        x = torch.randn(1, 3, cfg.d_model)
        cos, sin = self._cos_sin(rope, 3)
        out, _ = block(x, cos, sin)
        assert torch.allclose(out, x, atol=1e-5)

    def test_kv_cache_passthrough(self, cfg, block, rope):
        """Supplying a prior cache should increase seq_len in returned cache."""
        x = torch.randn(2, 4, cfg.d_model)
        cos, sin = self._cos_sin(rope, 4)
        _, cache = block(x, cos, sin, kv_cache=None)

        x2 = torch.randn(2, 1, cfg.d_model)
        cos2, sin2 = self._cos_sin(rope, 1, start_pos=4)
        _, cache2 = block(x2, cos2, sin2, kv_cache=cache)
        assert cache2.seq_len == 5

    def test_no_nan_in_output(self, cfg, block, rope):
        x = torch.randn(2, 5, cfg.d_model)
        cos, sin = self._cos_sin(rope, 5)
        out, _ = block(x, cos, sin)
        assert not torch.isnan(out).any()


class TestTransformerModel:
    def test_construction_does_not_raise(self, cfg):
        torch.manual_seed(0)
        TransformerModel(cfg)

    def test_param_count_positive(self, model):
        assert model.count_parameters() > 0

    def test_weight_tying(self, model):
        """token_emb.weight must be the exact same tensor used for unembedding."""
        emb_ptr = model.token_emb.weight.data_ptr()
        # If tying works correctly, there should be exactly one vocab-sized weight tensor.
        embedding_params = [
            p for p in model.parameters() if p.shape == (model.cfg.vocab_size, model.cfg.d_model)
        ]
        assert len(embedding_params) == 1, (
            "Expected exactly one vocab-size weight tensor (tied). "
            f"Found {len(embedding_params)}."
        )
        assert embedding_params[0].data_ptr() == emb_ptr

    def test_forward_output_shape(self, cfg, model):
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, caches = model(ids)
        assert logits.shape == (2, 8, cfg.vocab_size)

    def test_forward_returns_n_layer_caches(self, cfg, model):
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        _, caches = model(ids)
        assert len(caches) == cfg.n_layer

    def test_all_caches_are_kvcache(self, cfg, model):
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        _, caches = model(ids)
        assert all(isinstance(c, KVCache) for c in caches)

    def test_prefill_then_decode_shapes(self, cfg, model):
        ids = torch.randint(0, cfg.vocab_size, (1, 6))
        logits, caches = model(ids)
        assert logits.shape == (1, 6, cfg.vocab_size)

        next_id = torch.randint(0, cfg.vocab_size, (1, 1))
        logits2, caches2 = model(next_id, kv_caches=caches)
        assert logits2.shape == (1, 1, cfg.vocab_size)
        assert len(caches2) == cfg.n_layer

    def test_decode_cache_grows_correctly(self, cfg, model):
        ids = torch.randint(0, cfg.vocab_size, (1, 5))
        _, caches = model(ids)
        assert caches[0].seq_len == 5

        next_id = torch.randint(0, cfg.vocab_size, (1, 1))
        _, caches2 = model(next_id, kv_caches=caches)
        assert caches2[0].seq_len == 6

    def test_kv_cache_dtype_stays_bfloat16_during_decode(self, cfg, model):
        """
        End-to-end regression: after a prefill with bfloat16 inputs,
        the KV cache must remain bfloat16, not float32.
        """
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            _, caches = model(ids)
        for i, cache in enumerate(caches):
            assert (
                cache.k.dtype == torch.bfloat16
            ), f"Layer {i} KV cache k is {cache.k.dtype}, expected bfloat16"
            assert (
                cache.v.dtype == torch.bfloat16
            ), f"Layer {i} KV cache v is {cache.v.dtype}, expected bfloat16"

    def test_wrong_kv_cache_length_raises(self, cfg, model):
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        _, caches = model(ids)
        wrong_caches = caches[:-1]  # one too short
        with pytest.raises(ValueError, match="kv_caches"):
            model(ids, kv_caches=wrong_caches)

    def test_no_nan_in_logits(self, cfg, model):
        ids = torch.randint(0, cfg.vocab_size, (2, 6))
        logits, _ = model(ids)
        assert not torch.isnan(logits).any()

    def test_logits_not_all_identical(self, cfg, model):
        """Sanity: logits for different positions should differ."""
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        logits, _ = model(ids)
        assert not torch.allclose(logits[0, 0], logits[0, 1])

    def test_scale_residual_init_applied(self, cfg):
        """
        o_proj and down_proj must have smaller std than base layers.
        Expected residual std = 0.02 / sqrt(2 * n_layer).
        """
        torch.manual_seed(0)
        m = TransformerModel(cfg)
        base_std = 0.02

        for block in m.blocks:
            o_std = block.attn.o_proj.weight.std().item()
            d_std = block.ffn.down_proj.weight.std().item()

            # Residual layers should be noticeably smaller than base.
            # Allow 3x tolerance for random variation with small tensors.
            assert o_std < base_std * 0.8, (
                f"o_proj std={o_std:.5f} too close to base std={base_std} — "
                "SCALE_RESIDUAL_INIT may not have been applied"
            )
            assert (
                d_std < base_std * 0.8
            ), f"down_proj std={d_std:.5f} too close to base std={base_std}"

    def test_gradient_flows_through_full_model(self, cfg, model):
        ids = torch.randint(0, cfg.vocab_size, (2, 4))
        logits, _ = model(ids)
        loss = logits.sum()
        loss.backward()
        # Check gradients exist on a sample of leaf params
        assert model.token_emb.weight.grad is not None
        assert model.final_norm.weight.grad is not None
        assert model.blocks[0].attn.q_proj.weight.grad is not None

    def test_eval_mode_is_deterministic(self, cfg, model):
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 6))
        logits1, _ = model(ids)
        logits2, _ = model(ids)
        assert torch.allclose(logits1, logits2)
        model.train()

    def test_multi_decode_steps_accumulate_correctly(self, cfg, model):
        """Run 5 greedy decode steps and verify cache grows monotonically."""
        ids = torch.randint(0, cfg.vocab_size, (1, 3))
        _, caches = model(ids)

        for step in range(1, 6):
            next_id = torch.randint(0, cfg.vocab_size, (1, 1))
            _, caches = model(next_id, kv_caches=caches)
            assert caches[0].seq_len == 3 + step
