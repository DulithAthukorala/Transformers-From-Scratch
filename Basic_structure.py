import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


class PositionEncoding(nn.Module):
    def __init__(self, d_model=2, max_len=6):  # Embedding size = 2 , token length = 6
        super().__init__() # initialize the parent class

        pe = torch.zeros(max_len, d_model)  # postion encoding matrix of size (6, 2)

        position = torch.arange(start=0, end=max_len).unsqueeze(1)  # position vector of size (6,1)
        embedding_index = torch.arange(start=0, end=d_model, step=2)  # step 2 bcz, in the power of 2i anyways its multiples of 2

        div_term =  1/torch.tensor(10000.0)**(embedding_index/d_model)  # denominator term

        pe[:, 0::2] = torch.sin(position * div_term)  # even index positon 0,2,4
        pe[:, 1::2] = torch.cos(position * div_term)  # odd index position 1,3,5

        self.register_buffer('pe', pe)  # register buffer so that it is not considered as a model parameter 


    def forward(self, word_embedding):
        return word_embedding + self.pe[:word_embedding.size(0), :] # Rows from 0 up to L-1 , all columns  # word_embedding.size(0) = max_len
        

class Attention(nn.Module):
    def __init__(self, d_model=2):
        super().__init__()
        
        self.w_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)  # weight matrix for query
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)  # weight matrix for key
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)  # weight matrix for value

        self.row_dim = 0 
        self.col_dim = 1

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        q = self.w_q(encodings_for_q)  # (L, d_model)
        k = self.w_k(encodings_for_k)  # (L, d_model)
        v = self.w_v(encodings_for_v)  # (L, d_model)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim)) 

        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)  # scale by sqrt(d_k)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)  # fill masked positions with large negative value

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)  # softmax along columns (but works row-wise)

        attention_scores = torch.matmul(attention_percents, v) # (L, d_model)

        return attention_scores


class DecoderOnlyTransformer(L.LightningModule):
    def __init__(self, num_tokens=4, d_model=2, max_len=6): #num_tokens=how many unique tokens in the vocabulary
        super().__init__()

        self.we = nn.Embedding(num_embeddings=num_tokens, embedding_dim=d_model)  # create a lookup table for word embeddings (num_tokens, d_model)
        self.pe = PositionEncoding(max_len=max_len,d_model=d_model)  # position info for each token/word (max_len, d_model)
        self.self_attention = Attention(d_model=d_model)  # 

        self.fc = nn.Linear(in_features=d_model, out_features=num_tokens)  # final linear layer to project to vocab size

        self.loss = nn.CrossEntropyLoss()  # cross entropy loss for classification

    def forward(self, token_ids): # token_id -> where the token in the vocabulary
        word_embeddings = self.we(token_ids) 
        positional_encoded = self.pe(word_embeddings)

        mask = torch.tril(torch.ones((token_ids.size(dim=0), token_ids.size(dim=0))))
        mask = mask == 0  # convert to boolean mask


        