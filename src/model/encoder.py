"""
Encoder module consisting of multiple Transformer blocks.

Input X -> word embedding + positional embedding -> dropout and then passed through several
Transformer layers which do - >(Attention + ADD & Norm + Feed Forward + ADD & Norm)
"""

import math
import torch
import torch.nn as nn

from src.model.transformer_blocks import TransformerBlock


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size, # Unique Tokens in source vocabulary
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
        pre_norm=True,
    ):
        super().__init__()

        self.embed_size = embed_size
        self.device = device

        self.word_embedding = nn.Embedding(src_vocab_size, embed_size) # create a lookup table for word embeddings (src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size) # position info for each token/word (max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    pre_norm=pre_norm,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if x.dim() != 2:
            raise ValueError(f"Expected x shape (N: batch size, L: sequence length). Got {tuple(x.shape)}")

        N, seq_length = x.shape
        device = x.device # Locate the X tensor on(cpu or gpu)

        positions = torch.arange(0, seq_length, device=device).unsqueeze(0).expand(N, seq_length)

        out = self.word_embedding(x) * math.sqrt(self.embed_size) # Make token embeddings big positional information doesn’t drown them out
        out = out + self.position_embedding(positions)
        out = self.dropout(out)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
