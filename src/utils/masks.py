"""
We create 2 masks used by the 2017 Transformer:

1) Padding mask: blocks attention TO <pad> tokens in the KEY positions.
2) Causal mask: blocks attention to future tokens (decoder self-attention).

So after softmax, disallowed positions become ~0 probability.
"""

from __future__ import annotations

import torch


def make_padding_mask(token_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Build a padding mask for attention.

    Plain English:
    - token_ids contains <pad> tokens used only for batching.
    - We want the model to IGNORE those pad tokens when attending.

    This mask is intended to mask KEY positions (the last dimension in attention scores).

    Input:
    - token_ids: (B, L) int tensor of token IDs
    - pad_id:    integer ID used for <pad>

    Output:
    - mask: (B, 1, 1, L) float tensor
      where mask[b, 0, 0, j] = 0.0     if token j is NOT pad
                                   = -inf    if token j IS pad

    Why shape (B, 1, 1, L)?
    - Attention scores are (B, H, L_q, L_k)
    - This mask broadcasts across:
        - heads (H)
        - query positions (L_q)
      and only targets key positions (L_k = L).
    """
    if token_ids.dim() != 2:
        raise ValueError(f"token_ids must have shape (B, L). Got {tuple(token_ids.shape)}")

    # token_ids: (B, L)
    # is_pad: (B, L) boolean (True where pad)
    is_pad = token_ids.eq(pad_id)

    # We want float mask with 0 for allowed and -inf for disallowed.
    # mask_2d: (B, L)
    mask_2d = torch.zeros_like(token_ids, dtype=torch.float32)
    mask_2d = mask_2d.masked_fill(is_pad, float("-inf"))

    # Expand to (B, 1, 1, L)
    mask = mask_2d[:, None, None, :]

    return mask


def make_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Build a causal (look-ahead) mask for decoder self-attention.

    Plain English:
    - In decoder self-attention, token i must not attend to tokens > i (the future).
    - So we block the upper triangle above the diagonal.

    Input:
    - seq_len: L (an int)

    Output:
    - mask: (1, 1, L, L) float tensor
      where mask[0,0,i,j] = 0.0    if j <= i (allowed: past + present)
                           = -inf  if j > i  (disallowed: future)

    Why shape (1, 1, L, L)?
    - Attention scores are (B, H, L_q, L_k).
    - For decoder self-attention, L_q = L_k = L.
    - This mask broadcasts across batch (B) and heads (H).
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive. Got {seq_len}")

    # Start with zeros: (L, L)
    mask_2d = torch.zeros((seq_len, seq_len), dtype=torch.float32, device=device)

    # upper_triangle (above diagonal) should be -inf
    # torch.triu(..., diagonal=1) picks entries where j > i
    upper = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
    mask_2d = mask_2d.masked_fill(upper, float("-inf"))

    # Expand to (1, 1, L, L)
    return mask_2d[None, None, :, :]


def combine_masks(
    padding_mask: torch.Tensor | None,
    causal_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """
    Combine padding and causal masks by addition.

    Because masks are additive (0 or -inf), we can just add them:
    - If either mask disallows a position -> -inf wins.
    - If both allow -> 0 stays.

    padding_mask expected shape: (B, 1, 1, L_k)
    causal_mask expected shape:  (1, 1, L_q, L_k)  (decoder self-attn: L_q=L_k)

    Output:
    - combined mask broadcastable to (B, H, L_q, L_k), or None if both None.
    """
    if padding_mask is None and causal_mask is None:
        return None
    if padding_mask is None:
        return causal_mask
    if causal_mask is None:
        return padding_mask
    return padding_mask + causal_mask
