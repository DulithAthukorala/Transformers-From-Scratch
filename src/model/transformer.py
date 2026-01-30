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

        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            device=self.device,
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
            device=self.device,
            max_length=max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src: torch.Tensor):
        # (N, 1, 1, src_len) with True = keep, False = block
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg: torch.Tensor):
        N, trg_len = trg.shape

        # causal: (1, 1, trg_len, trg_len)
        causal = torch.tril(torch.ones((trg_len, trg_len), device=trg.device, dtype=torch.bool))
        causal = causal.unsqueeze(0).unsqueeze(0)

        # pad: (N, 1, 1, trg_len)
        pad = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # combined: (N, 1, trg_len, trg_len)
        return causal & pad

    def forward(self, src: torch.Tensor, trg: torch.Tensor):
        if src.dim() != 2 or trg.dim() != 2:
            raise ValueError(f"src and trg must be (N, L). Got src{tuple(src.shape)}, trg{tuple(trg.shape)}")

        src = src.to(self.device)
        trg = trg.to(self.device)

        src_mask = self.make_src_mask(src)  # bool mask
        trg_mask = self.make_trg_mask(trg)  # bool mask

        enc_out = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_out, src_mask, trg_mask)
        return out
