import torch
import torch.nn as nn

from src.model.encoder import Encoder
from src.model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        src_pad_idx: int,
        trg_pad_idx: int,
        embed_size: int = 512,
        num_layers: int = 6,
        forward_expansion: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
        device="cpu",
        max_length: int = 100,
    ):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            device=device,
            forward_expansion=forward_expansion,
            dropout=dropout,
            max_length=max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size=trg_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            device=device,
            max_length=max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src: torch.Tensor):
        # 1 = keep, 0 = mask out (matches masked_fill(mask == 0, ...))
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg: torch.Tensor):
        N, trg_len = trg.shape

        causal = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).expand(
            N, 1, trg_len, trg_len
        )  # (N,1,trg_len,trg_len)

        pad = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)  # (N,1,1,trg_len)

        return causal * pad  # still 1s and 0s, matches mask == 0 logic

    def forward(self, src: torch.Tensor, trg: torch.Tensor):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_out = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_out, src_mask, trg_mask)
        return out
