import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        if self.head_dim * heads != embed_size:
            raise ValueError("embed_size must be divisible by heads")

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        queries: torch.Tensor,
        mask: torch.Tensor | None,
    ):
        N = queries.shape[0]
        value_len = values.shape[1]
        key_len = keys.shape[1]
        query_len = queries.shape[1]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)

        if mask is not None:
            energy = energy.masked_fill(~mask, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=-1)

        out = torch.einsum("nhqk,nkhd->nqhd", attention, values)
        out = out.reshape(N, query_len, self.heads * self.head_dim)

        out = self.fc_out(out)
        return out
