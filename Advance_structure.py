import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size  # word embedding size (one word = 1 dimensional vector) , (sentence = multiple words = 2D matrix) , (batch of sentences = 3D matrix)
        self.heads = heads # number of heads (parallel attention layers)
        self.head_dim = embed_size // heads # dimension of each head

        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)