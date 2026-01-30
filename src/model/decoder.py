import math
import torch
import torch.nn as nn

from src.model.transformer_blocks import DecoderBlock


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
        pre_norm=True,
    ):
        super().__init__()

        self.embed_size = embed_size
        self.device = device

        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size=embed_size,
                    heads=heads,
                    forward_expansion=forward_expansion,
                    dropout=dropout,
                    pre_norm=pre_norm,
                )
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, trg_mask=None):
        if x.dim() != 2:
            raise ValueError(f"Expected x shape (N, L). Got {tuple(x.shape)}")

        N, seq_length = x.shape
        device = x.device

        positions = torch.arange(0, seq_length, device=device).unsqueeze(0).expand(N, seq_length)

        x = self.word_embedding(x) * math.sqrt(self.embed_size)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out
