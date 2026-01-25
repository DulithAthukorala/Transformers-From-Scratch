import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import lightning as L

token_to_id = {
    "what": 0,
    "is": 1,
    "statquest": 2,
    "<EOS>": 3,
    "awesome": 4
}
id_to_token = dict(map(reversed, token_to_id.items()))

inputs = torch.tensor([[token_to_id["what"], ## input #1: what is statquest <EOS> awesome
                        token_to_id["is"], 
                        token_to_id["statquest"], 
                        token_to_id["<EOS>"],
                        token_to_id["awesome"]], 
                       
                       [token_to_id["statquest"], # input #2: statquest is what <EOS> awesome
                        token_to_id["is"], 
                        token_to_id["what"], 
                        token_to_id["<EOS>"], 
                        token_to_id["awesome"]]])

labels = torch.tensor([[token_to_id["is"], 
                        token_to_id["statquest"], 
                        token_to_id["<EOS>"], 
                        token_to_id["awesome"], 
                        token_to_id["<EOS>"]],  
                       
                       [token_to_id["is"], 
                        token_to_id["what"], 
                        token_to_id["<EOS>"], 
                        token_to_id["awesome"], 
                        token_to_id["<EOS>"]]])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

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


    def forward(self, word_embeddings):
        return word_embeddings + self.pe[:word_embeddings.size(0), :] # Rows from 0 up to L-1 , all columns  # word_embedding.size(0) = max_len
        

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

        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)  # final linear layer to project to vocab size

        self.loss = nn.CrossEntropyLoss()  # cross entropy loss for classification

    def forward(self, token_ids): # token_id -> where the token in the vocabulary
        word_embeddings = self.we(token_ids) 
        positional_encoded = self.pe(word_embeddings)

        mask = torch.tril(torch.ones((token_ids.size(dim=0), token_ids.size(dim=0))))
        mask = mask == 0  # convert to boolean mask

        self_attention_values = self.self_attention(positional_encoded, positional_encoded, positional_encoded, mask=mask)

        residual_connection_values = positional_encoded + self_attention_values

        fc_layer_output = self.fc_layer(residual_connection_values)

        return fc_layer_output

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch
        outputs = self.forward(input_tokens)
        loss = self.loss(outputs,labels[0])

        return loss


model = DecoderOnlyTransformer(num_tokens=len(token_to_id), d_model=2, max_len=6)
trainer = L.Trainer(max_epochs=30)
trainer.fit(model, train_dataloaders=dataloader)

model_input = torch.tensor([
    token_to_id["what"],
    token_to_id["is"],
    token_to_id["statquest"],
    token_to_id["<EOS>"]
])

input_length = model_input.size(dim=0)

predictions = model(model_input)
predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
predicted_ids = predicted_id

max_length = 6
for i in range(input_length, max_length):
    if predicted_id == token_to_id["<EOS>"]:
        break

    model_input = torch.cat((model_input, predicted_id))

    predictions = model(model_input)
    predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
    predicted_ids = torch.cat((predicted_ids, predicted_id))

print("Predicted Tokens:\n")
for id in predicted_ids:
    print("\t", id_to_token[id.item()])



        