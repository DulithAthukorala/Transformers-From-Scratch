import torch
import torch.nn as nn

from src.model.attention import SelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, embed_size: int, heads: int, dropout: float, forward_expansion: int):
        super().__init__()

        self.attention = SelfAttention(embed_size, heads)  # normal (or masked if mask passed)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        value: torch.Tensor,
        key: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attention = self.attention(value, key, query, mask)
        x = self.norm1(query + self.dropout(attention))

        forward = self.feed_forward(x)
        out = self.norm2(x + self.dropout(forward))

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size: int, heads: int, forward_expansion: int, dropout: float):
        super().__init__()

        self.self_attention = SelfAttention(embed_size, heads)  # masked via trg_mask
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        value: torch.Tensor,
        key: torch.Tensor,
        src_mask: torch.Tensor | None,
        trg_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attention = self.self_attention(x, x, x, trg_mask)
        query = self.norm(x + self.dropout(attention))

        out = self.transformer_block(value, key, query, src_mask)
        return out
