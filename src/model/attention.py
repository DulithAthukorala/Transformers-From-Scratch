from __future__ import annotations

import math
import torch


class ScaledDotProductAttention:
    """
    Implements:
        softmax((QK^T)/sqrt(d_k) + mask) V

    Expected shapes:
      Q: (B, H, Lq, d_k)
      K: (B, H, Lk, d_k)
      V: (B, H, Lv, d_k)
      mask (optional): broadcastable to (B, H, Lq, Lk)

    B = batch size
    H = number of heads
    Lq / Lk / Lv = query/key/value sequence length
    d_k = head dimension
      
    """

    def __call__(self, Q, K, V, mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        if Q.dim() != 4 or K.dim() != 4 or V.dim() != 4:
            raise ValueError(f"Q,K,V must be 4D tensors. Got Q{tuple(Q.shape)}, K{tuple(K.shape)}, V{tuple(V.shape)}")

        Bq, Hq, Lq, dk_q = Q.shape
        Bk, Hk, Lk, dk_k = K.shape
        Bv, Hv, Lv, dk_v = V.shape

        if (Bq, Hq) != (Bk, Hk) or (Bq, Hq) != (Bv, Hv):
            raise ValueError("Batch/head dims must match for Q,K,V.")

        if dk_q != dk_k or dk_q != dk_v:
            raise ValueError("Last Head dim (d_k) must match for Q,K,V.")

        if Lk != Lv: 
            '''
            queries can ask about a different sequence,
            but Lk must equal Lv because every key must have a value to read from.
            '''
            raise ValueError("K and V must have same sequence length (L_k).") 

        d_k = dk_q  # head dimension (bcz all are same)

        # --- 1) compute raw attention scores ---
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)

        # --- 2) apply mask (if provided) ---
        if mask is not None:
            scores = scores + mask

        # --- 3) softmax over keys (last dimension) ---
        attn = torch.softmax(scores, dim=-1)

        # --- 4) weighted sum of values ---
        out = attn @ V

        # out: (B, H, L_q, d_k) --> for the model
        # attn: (B, H, L_q, L_k) --> for visualization / analysis purposes
        return out, attn
