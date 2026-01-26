import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size  # word embedding size (how many numbers represent a word)
        self.heads = heads # number of heads (parallel attention layers)
        self.embedding_numbers_per_head = embed_size // heads # features per head

        assert (self.embedding_numbers_per_head * heads == embed_size), "Embedding size needs to be divisible by heads"

        self.queries = nn.Linear(self.embedding_numbers_per_head, self.embedding_numbers_per_head, bias=False) # nn.Linear -> initialize random weight matrix
        self.keys = nn.Linear(self.embedding_numbers_per_head, self.embedding_numbers_per_head, bias=False)
        self.values = nn.Linear(self.embedding_numbers_per_head, self.embedding_numbers_per_head, bias=False)
        self.fc_out = nn.Linear(heads * self.embedding_numbers_per_head, embed_size)  # final output linear layer

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]  # (B, seq_length, embed_size) -> B (how many sequences we have in a batch)
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1] # (B, seq_length, embed_size) -> seq_length

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.embedding_numbers_per_head)
        keys = keys.reshape(N, key_len, self.heads, self.embedding_numbers_per_head)
        queries = queries.reshape(N, query_len, self.heads, self.embedding_numbers_per_head)

        energy = torch.einsum("nqhe,nkhe->nhqk", [queries, keys])  # Einstein summation for matrix multiplication
        # queries shape: (N, query_len, heads, embedding_numbers_per_head)
        # keys shape: (N, key_len, heads, embedding_numbers_per_head)
        # energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))  # fill masked positions with large negative value

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # softmax along the last dimension (key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.embedding_numbers_per_head
        ) 
        # attention shape: (N, heads, query_len, key_len)  
        # values shape: (N, value_len, heads, embedding_numbers_per_head)
        # after einsum: (N, query_len, heads, embedding_numbers_per_head) -> then flatten dimensions

        out = self.fc_out(out)  # final linear layer
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add & Norm
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
