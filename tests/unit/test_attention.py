import torch

from model.attention import _build_prefill_mask


def test_build_prefill_mask():
    """Case-3 mask exact correctness: compare against a hand-built mask."""
    q_len = 3
    cache_len = 2
    device = torch.device("cpu")

    mask = _build_prefill_mask(q_len, cache_len, device)

    # Expected mask for q_len=3, cache_len=2:
    # Query 0 can attend to Cache 0, 1 and itself (Query 0)
    # Query 1 can attend to Cache 0, 1, Query 0, and itself (Query 1)
    # Query 2 can attend to everything.
    expected = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
        ],
        dtype=torch.bool,
        device=device,
    )

    assert torch.equal(mask, expected), f"Expected:\n{expected}\nGot:\n{mask}"
