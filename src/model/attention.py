"""
This module implements a multi-head self-attention mechanism With and Without the Masking.
    - Implemented in the Encoder as MultiHeadAttention block And
    - In Decoder as MaskedMultiHeadAttention and Cross/MultiHeadAttention Blocks
"""
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.0):
        """
        Token vector (512):
        |----64----|----64----|----64----|----64----|----64----|----64----|----64----|----64----|
             h1         h2         h3         h4         h5         h6         h7         h8
        """
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads  # word Embeddings Per Head

        if self.head_dim * heads != embed_size:
            raise ValueError("embed_size must be divisible by heads")

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.attn_dropout = nn.Dropout(dropout) # Dropout Regularization rate

    def forward(self, values, keys, queries, mask=None, return_attention=False):
        """
        mostly values, keys, queries are the input tensor
        But in cross attention, queries come from decoder, keys and values come from encoder, 
        in order to handle that values, keys, queries are passed separately

        values:  (N, value_len, embed_size)
        keys:    (N, key_len, embed_size)
        queries: (N, query_len, embed_size)
        """

        N = queries.shape[0]
        value_len = values.shape[1]
        key_len = keys.shape[1]
        query_len = queries.shape[1]

        if values.shape[0] != N or keys.shape[0] != N:
            raise ValueError("Batch size mismatch between values/keys/queries")

        # (N, len, embed_size)
        # To Avoid Q=K=V
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # (N, len, heads, head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # energy: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys) # Compute Q·K^T

        # mask should be broadcastable to (N, 1 or heads, query_len, key_len)
        # Our convention: mask=True keep, mask=False block
        if mask is not None:
            energy = energy.masked_fill(~mask, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=-1)
        attention = self.attn_dropout(attention)

        # out: (N, len, heads, head_dim)
        out = torch.einsum("nhqk,nkhd->nqhd", attention, values)  # (N, query_len, heads, head_dim)
        out = out.reshape(N, query_len, self.heads * self.head_dim)
        
        out = self.fc_out(out)
        if return_attention: # For Development Purposes
            return out, attention

        return out
