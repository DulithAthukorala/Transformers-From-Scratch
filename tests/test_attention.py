import torch

from src.model.attention import SelfAttention


def test_attention_weights_sum_to_one():
    torch.manual_seed(0)

    attn = SelfAttention(embed_size=32, heads=4, dropout=0.0)

    N, Lq, Lk, E = 2, 3, 5, 32
    q = torch.randn(N, Lq, E)
    k = torch.randn(N, Lk, E)
    v = torch.randn(N, Lk, E)

    out, weights = attn(v, k, q, mask=None, return_attention=True)

    assert out.shape == (N, Lq, E)
    assert weights.shape == (N, 4, Lq, Lk)

    # For each (N, head, query), probabilities over keys sum to ~1
    s = weights.sum(dim=-1)
    ones = torch.ones_like(s)
    assert torch.allclose(s, ones, atol=1e-5)


def test_masked_keys_get_zero_probability():
    torch.manual_seed(0)

    attn = SelfAttention(embed_size=32, heads=4, dropout=0.0)

    N, Lq, Lk, E = 1, 2, 4, 32
    q = torch.randn(N, Lq, E)
    k = torch.randn(N, Lk, E)
    v = torch.randn(N, Lk, E)

    # mask blocks the last 2 keys
    # shape: (N, 1, 1, Lk) broadcastable to (N, heads, Lq, Lk)
    mask = torch.tensor([[[[True, True, False, False]]]])

    out, weights = attn(v, k, q, mask=mask, return_attention=True)

    assert out.shape == (N, Lq, E)

    # weights on masked keys should be ~0
    masked_part = weights[..., 2:]  # last two keys
    assert torch.all(masked_part < 1e-6).item() is True
