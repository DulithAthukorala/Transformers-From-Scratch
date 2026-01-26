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

