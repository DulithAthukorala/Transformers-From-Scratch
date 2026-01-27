import torch
import torch.nn as nn


class SelfAttention(nn.Module):  # Does the normal/masked Multi-Head Self-Attention
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
        ''' 
        mostly values, keys, queries are the input tensor
        But in cross attention, queries come from decoder, keys and values come from encoder, in order to handle that values, keys, queries are passed separately

        '''
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

class TransformerBlock(nn.Module): # Attention + add & norm + feedforward + add & norm
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
        # Attention
        attention = self.attention(value, key, query, mask)
        # Add & Norm
        x = self.dropout(self.norm1(attention + query)) # x = self.norm1(query + self.dropout(attention))
        # Feed Forward
        forward = self.feed_forward(x)
        # Add & Norm
        out = self.dropout(self.norm2(forward + x)) # out = self.norm2(x + self.dropout(forward))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size, # how many unique words in the vocabulary
        embed_size,  # how many numbers represent a word/token
        num_layers, # number of transformer block layers (Layers learn from previous layer's output, syntax -> phrase structure -> meaning)
        heads, # heads learn different stuff from the same senetence
        device, 
        forward_expansion, # Feedforward hidden layer size
        dropout, # dropout rate
        max_length, # maximum length of input sentence (for positional encoding)
    ):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)  # create a lookup table for word embeddings (src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)  # position info for each token/word (max_length, embed_size) # use stat Q 
        self.layers = nn.ModuleList( # ModuleList = “tell PyTorch these are real layers”
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
            ]
            for _ in range(num_layers)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)  # create position tensor

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))  # add word embeddings and position embeddings

        for layer in self.layers:
            out = layer(out, out, out, mask) # In Encoder: value, key, query are all the same

        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super().__init__()
        self.self_attention = SelfAttention(embed_size, heads) # Masked Multi-Head Self-Attention
        self.norm = nn.LayerNorm(embed_size) # Layer Normalization
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, value, key, src_mask, trg_mask):
        # Masked Multi-Head Self-Attention
        attention = self.self_attention(x, x, x, trg_mask)
        x = self.dropout(self.norm(attention + x))

        # Encoder-Decoder Attention + Feedforward + Add & Norm
        out = self.transformer_block(value, key, x, src_mask)
        return out
    
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
    ):
        super().__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out
    

