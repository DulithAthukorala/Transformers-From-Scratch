from __future__ import annotations

import torch
import torch.nn as nn

from src.model.attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Input:
      x_q: (B, L_q, emb_size)
      x_k: (B, L_k, emb_size)
      x_v: (B, L_v, emb_size)

    Output:
      out:  (B, L_q, emb_size)

    B = batch size
    L_q / L_k / L_v = query/key/value sequence length
    emb_size = embedding size

    """

    def __init__(self, emb_size, num_heads):
        super().__init__()

        if emb_size % num_heads != 0:
            raise ValueError(f"Embedding size ({emb_size}) must be divisible by number of heads ({num_heads})")

        self.emb_size = emb_size 
        self.num_heads = num_heads
        self.d_k = emb_size // num_heads # dimension per head

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(emb_size, emb_size)
        self.W_k = nn.Linear(emb_size, emb_size)
        self.W_v = nn.Linear(emb_size, emb_size)

        # Output projection
        self.W_o = nn.Linear(emb_size, emb_size)

        self.attention = ScaledDotProductAttention()



    def _split_heads(self, x):
        B, L, _ = x.shape
        x = x.view(B, L, self.num_heads, self.d_k)  # (B, L, H, d_k)

        return x.transpose(1, 2) # (B, H, L, d_k)

    def _combine_heads(self, x):
        B, H, L, d_k = x.shape
        x = x.transpose(1, 2)  # (B, L, H, d_k)
        
        return x.contiguous().view(B, L, H * d_k)  # (B, L, emb_size)



    def forward(self, x_q, x_k, x_v, mask):
        """
        Forward pass for multi-head attention.
        """
        # 1) Linear projections
        Q = self.W_q(x_q)
        K = self.W_k(x_k)
        V = self.W_v(x_v)

        # 2) Split into heads
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # 3) Apply scaled dot-product attention
        out, _ = self.attention(Q, K, V, mask)

        # 4) Combine heads
        out = self._combine_heads(out)

        # 5) Final linear projection
        return self.W_o(out)
