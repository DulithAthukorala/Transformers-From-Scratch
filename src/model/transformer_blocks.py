"""
This module implements:
1. Transformer Block(Attention + ADD & Norm + Feed Forward + ADD & Norm)
2. Decoder Block (Masked Self-Attention + ADD & Norm + Cross-Attention + ADD & Norm + Feed Forward + ADD & Norm)
"""

import torch
import torch.nn as nn

from src.model.attention import SelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, pre_norm=True):
        super().__init__()

        self.pre_norm = pre_norm

        self.attention = SelfAttention(embed_size, heads, dropout=dropout)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        hidden = forward_expansion * embed_size # Hidden layer size in Feed Forward network
        self.fc1 = nn.Linear(embed_size, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, embed_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None, return_attention=False):
        # Pre-Norm variant (more stable version): LN -> sublayer -> residual
        if self.pre_norm:
            q = self.norm1(query) 
            if value.shape == query.shape and key.shape == query.shape:
                v = q
                k = q
            else:
                v = value
                k = key
            if return_attention:
                attn_out, attn = self.attention(v, k, q, mask, return_attention=True)
            else:
                attn_out = self.attention(v, k, q, mask)
                
            # ADD & Norm
            x = query + self.dropout(attn_out)
            y = self.norm2(x)

            # Feed Forward network + ADD & Norm
            ff = self.fc2(self.dropout(self.act(self.fc1(y)))) # Feed Forward network (512 -> 2048 -> 512)
            out = x + self.dropout(ff)

            if return_attention:
                return out, attn
            return out

        # Post-Norm variant (paper-implementation): sublayer -> residual -> LN
        if return_attention:
            attn_out, attn = self.attention(value, key, query, mask, return_attention=True)
        else:
            attn_out = self.attention(value, key, query, mask)

        x = self.norm1(query + self.dropout(attn_out))

        ff = self.fc2(self.dropout(self.act(self.fc1(x)))) # Feed Forward network (512 -> 2048 -> 512)
        out = self.norm2(x + self.dropout(ff))

        if return_attention:
            return out, attn
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, pre_norm=True):
        super().__init__()

        self.pre_norm = pre_norm

        self.self_attention = SelfAttention(embed_size, heads, dropout=dropout)

        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

        self.transformer_block = TransformerBlock(
            embed_size=embed_size,
            heads=heads,
            dropout=dropout,
            forward_expansion=forward_expansion,
            pre_norm=pre_norm,
        )

    def forward(self, x, value, key, src_mask=None, trg_mask=None):
        if self.pre_norm:
            q = self.norm(x)
            self_out = self.self_attention(q, q, q, trg_mask)
            query = x + self.dropout(self_out)
        else:
            self_out = self.self_attention(x, x, x, trg_mask)
            query = self.norm(x + self.dropout(self_out))

        out = self.transformer_block(value, key, query, src_mask)
        return out
