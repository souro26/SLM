import math

import torch

from model.config import ModelConfig
from model.transformer import TransformerModel


def test_scale_residual_init_fires():
    """Verify SCALE_RESIDUAL_INIT tag triggers a smaller std initialization."""
    cfg = ModelConfig(
        n_layer=4,
        d_model=256,
        n_heads_q=4,
        n_heads_kv=2,
        head_dim=64,
        d_ffn=1024,
        vocab_size=100,
        context_length=64,
    )

    # Model init happens in __init__, so we just instantiate
    torch.manual_seed(42)
    model = TransformerModel(cfg)

    expected_base_std = 0.02
    expected_residual_std = 0.02 / math.sqrt(2 * cfg.n_layer)

    # q_proj is not tagged with SCALE_RESIDUAL_INIT
    base_std = model.blocks[0].attn.q_proj.weight.std().item()
    # o_proj is tagged with SCALE_RESIDUAL_INIT
    resid_std = model.blocks[0].attn.o_proj.weight.std().item()

    # We allow some tolerance since it's a statistical measure,
    # but 256*256 elements is enough for std to be quite accurate.
    assert math.isclose(
        base_std, expected_base_std, rel_tol=0.1
    ), f"q_proj std {base_std} != {expected_base_std}"
    assert math.isclose(
        resid_std, expected_residual_std, rel_tol=0.1
    ), f"o_proj std {resid_std} != {expected_residual_std}"

    assert resid_std < base_std, "Residual layers should have a noticeably smaller init std"
