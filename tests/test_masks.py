"""
tests/test_masks.py

Run with:
  python -m pytest

These tests verify:
- shapes are correct
- -inf appears exactly where it should
- masks broadcast the way attention expects
"""

import torch

from src.utils.masks import make_padding_mask, make_causal_mask, combine_masks


def test_make_padding_mask_shape_and_values():
    # token_ids: (B=2, L=5)
    token_ids = torch.tensor(
        [
            [4, 7, 0, 0, 0],  # pads at positions 2,3,4
            [1, 2, 3, 4, 5],  # no pads
        ],
        dtype=torch.long,
    )
    pad_id = 0

    mask = make_padding_mask(token_ids, pad_id)

    # Expect shape (B, 1, 1, L)
    assert mask.shape == (2, 1, 1, 5)

    # First row: positions 2,3,4 should be -inf
    assert mask[0, 0, 0, 0].item() == 0.0
    assert mask[0, 0, 0, 1].item() == 0.0
    assert mask[0, 0, 0, 2].item() == float("-inf")
    assert mask[0, 0, 0, 3].item() == float("-inf")
    assert mask[0, 0, 0, 4].item() == float("-inf")

    # Second row: no pads, all zeros
    assert torch.all(mask[1] == 0.0)


def test_make_causal_mask_shape_and_values():
    L = 4
    mask = make_causal_mask(L)

    # Expect shape (1, 1, L, L)
    assert mask.shape == (1, 1, 4, 4)

    # Allowed: j <= i -> 0
    assert mask[0, 0, 0, 0].item() == 0.0
    assert mask[0, 0, 1, 0].item() == 0.0
    assert mask[0, 0, 2, 2].item() == 0.0

    # Disallowed: j > i -> -inf
    assert mask[0, 0, 0, 1].item() == float("-inf")
    assert mask[0, 0, 0, 2].item() == float("-inf")
    assert mask[0, 0, 1, 3].item() == float("-inf")


def test_combine_masks_broadcastable():
    # Batch=2, L=4
    token_ids = torch.tensor(
        [
            [9, 0, 8, 0],  # pads at 1 and 3
            [1, 2, 3, 4],  # no pads
        ],
        dtype=torch.long,
    )
    pad_mask = make_padding_mask(token_ids, pad_id=0)  # (2,1,1,4)
    causal = make_causal_mask(seq_len=4)               # (1,1,4,4)

    combined = combine_masks(pad_mask, causal)

    # combined should broadcast to (B, H, L, L) in attention
    assert combined.shape == (2, 1, 4, 4)

    # Check: future is masked everywhere (upper triangle)
    assert combined[0, 0, 0, 1].item() == float("-inf")

    # Check: padding key positions are masked for all query positions
    # For batch 0, key position 1 is pad => combined[:, :, :, 1] should be -inf across all query rows
    assert torch.all(combined[0, 0, :, 1] == float("-inf"))
    # For batch 1, no pads => key position 1 shouldn't be forced -inf (except by causal)
    assert combined[1, 0, 3, 1].item() == 0.0  # query=3 can attend to key=1 (past)
