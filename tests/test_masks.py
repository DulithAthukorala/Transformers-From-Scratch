import torch

from src.model.transformer import Transformer


def test_src_mask_shape_and_values():
    model = Transformer(
        src_vocab_size=10,
        trg_vocab_size=10,
        src_pad_idx=0,
        trg_pad_idx=0,
        embed_size=32,
        num_layers=1,
        heads=4,
        device="cpu",
        max_length=20,
    )

    src = torch.tensor([
        [1, 2, 0, 0],
        [3, 4, 5, 0],
    ])

    m = model.make_src_mask(src)

    assert m.shape == (2, 1, 1, 4)
    # keep = True, pad = False
    assert torch.equal(m[0, 0, 0], torch.tensor([True, True, False, False]))
    assert torch.equal(m[1, 0, 0], torch.tensor([True, True, True, False]))


def test_trg_mask_causal_and_padding():
    model = Transformer(
        src_vocab_size=10,
        trg_vocab_size=10,
        src_pad_idx=0,
        trg_pad_idx=0,
        embed_size=32,
        num_layers=1,
        heads=4,
        device="cpu",
        max_length=20,
    )

    trg = torch.tensor([
        [1, 2, 3, 0],   # last is pad
        [4, 5, 0, 0],   # last two are pad
    ])

    m = model.make_trg_mask(trg)

    assert m.shape == (2, 1, 4, 4)

    # Causal: position 0 cannot attend to 1,2,3
    assert m[0, 0, 0, 1].item() is False
    assert m[0, 0, 0, 2].item() is False
    assert m[0, 0, 0, 3].item() is False

    # Causal: position 2 can attend to 0,1,2 but not 3
    assert m[0, 0, 2, 0].item() is True
    assert m[0, 0, 2, 1].item() is True
    assert m[0, 0, 2, 2].item() is True
    assert m[0, 0, 2, 3].item() is False

    # Padding: no one should attend to padded key positions (last column(s))
    # For trg[0], index 3 is pad => that key column should be False for all queries
    assert torch.all(m[0, 0, :, 3] == False).item() is True

    # For trg[1], indices 2 and 3 are pad => those key columns should be False
    assert torch.all(m[1, 0, :, 2] == False).item() is True
    assert torch.all(m[1, 0, :, 3] == False).item() is True
