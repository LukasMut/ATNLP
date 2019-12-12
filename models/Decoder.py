__all__ = ['DecoderRNN', 'DecoderLSTM', 'DecoderGRU', 'AttnDecoderRNN', 'AttnDecoderLSTM', 'AttnDecoderGRU']

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Attention import *

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

torch.manual_seed(42)

class DecoderRNN(nn.Module):
    
    def __init__(self, emb_size:int, hidden_size:int, out_size:int, n_layers:int=2, dropout:float=0.5):
        super(DecoderRNN, self).__init__()
        self.emb_size = emb_size        
        self.hidden_size = hidden_size
        self.out_size = out_size # |V|
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(out_size, emb_size, padding_idx=0)
        #self.embedding_dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, out_size)
        
    def forward(self, x_batch, hidden):
        batch_size = x_batch.size(0)
        embedded = self.embedding(x_batch).view(batch_size, 1, -1)
        embedded = F.relu(embedded)
        out, hidden = self.rnn(embedded, hidden)
        # convert 3-dimensional tensor into 2-dimensional matrix (required for linear layer)
        out = out.squeeze(1)
        logits = self.linear(out)
        probas = F.softmax(logits, dim=1)
        return probas, hidden
    
    def init_hidden(self, batch_size:int=1):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.emb_size, device=device)
        return nn.init.xavier_uniform_(hidden_state)
    
    
class DecoderLSTM(nn.Module):
    
    def __init__(self, emb_size:int, hidden_size:int, out_size:int, n_layers:int=2, dropout:float=0.5):
        super(DecoderLSTM, self).__init__()
        self.emb_size = emb_size        
        self.hidden_size = hidden_size
        self.out_size = out_size # |V|
        self.n_layers = n_layers
        self.dropout = dropout
        self.emb_size = emb_size
        
        self.embedding = nn.Embedding(out_size, emb_size, padding_idx=0)
        #self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, out_size)
        
    def forward(self, x_batch, hidden):
        batch_size = x_batch.size(0)
        embedded = self.embedding(x_batch).view(batch_size, 1, -1)
        embedded = F.relu(embedded)
        out, hidden = self.lstm(embedded, hidden)
        # convert 3-dimensional tensor into 2-dimensional matrix (required for linear layer)
        out = out.squeeze(1)
        logits = self.linear(out)
        probas = F.softmax(logits, dim=1)
        return probas, hidden
    
    def init_hidden(self, batch_size:int=1):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.emb_size, device=device)
        cell_state = torch.zeros(self.n_layers, batch_size, self.emb_size, device=device)
        hidden = (nn.init.xavier_uniform_(hidden_state), nn.init.xavier_uniform_(cell_state))
        return hidden
    
class DecoderGRU(nn.Module):
    
    def __init__(self, emb_size:int, hidden_size:int, out_size:int, n_layers:int=2, dropout:float=0.5, max_length:bool=None):
        super(DecoderGRU, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.out_size = out_size # |V|
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(out_size, emb_size, padding_idx=0)
        self.gru = nn.GRU(emb_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, out_size)
        
    def forward(self, x_batch, hidden):
        batch_size = x_batch.size(0)
        # NOTE: first dim represents batch size, second represents sequence length, third dim embedding size (if batch_first=True)
        embedded = self.embedding(x_batch).view(batch_size, 1, -1)
        embedded = F.relu(embedded) #TODO: figure out, whether applying a ReLu on embedding inputs is useful
        out, hidden = self.gru(embedded, hidden)
        # convert 3-dimensional tensor into 2-dimensional matrix (required for linear layer)
        out = out.squeeze(1)
        logits = self.linear(out)
        probas = F.softmax(logits, dim=1)
        return probas, hidden
    
    def init_hidden(self, batch_size:int=1):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.emb_size, device=device)
        return nn.init.xavier_uniform_(hidden_state)
    
    
class AttnDecoderRNN(nn.Module):
    
    def __init__(self, emb_size:int,  hidden_size:int, out_size:int, n_layers:int, dropout:float,
                 attention_version:str, max_length:bool=None):
        super(AttnDecoderRNN, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.out_size = out_size # |V|
        self.n_layers = n_layers
        self.dropout = dropout
        self.attention_version = attention_version
        self.max_length = max_length # max source sequence length
        
        self.embedding = nn.Embedding(out_size, emb_size, padding_idx=0)
        
        if attention_version == 'general':
            isinstance(max_length, int), 'General attention requires maximum command sequence length'
            self.attention = GeneralAttention(emb_size, hidden_size, max_length)
            self.linear = nn.Linear(hidden_size, out_size)
            self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        elif attention_version == 'multiplicative':
            self.attention = MultiplicativeAttention(hidden_size)
            self.linear = nn.Linear(hidden_size * 2, out_size)
            self.rnn = nn.RNN(emb_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError("Attention version must be one of ['general', 'multiplicative']")

        
    def forward(self, x_batch, hidden, encoder_hiddens):
        batch_size = x_batch.size(0)
        # NOTE: first dim represents batch size, second represents sequence length, third dim embedding size (if batch_first=True)
        embedded = self.embedding(x_batch).view(batch_size, 1, -1)
        embedded = F.relu(embedded)
        
        if self.attention_version == 'multiplicative':
            out, hidden = self.rnn(embedded, hidden)
            out = F.relu(out)
            context, attn_weights = self.attention(hidden[-1], encoder_hiddens)
            logits = self.linear(torch.cat((out, context), 2).squeeze(1))
        elif self.attention_version == 'general':
            context, attn_weights = self.attention(embedded, hidden[0], encoder_hiddens)
            out = F.relu(context)
            out, hidden = self.rnn(out, hidden)
            logits = self.linear(out.squeeze(1))
            
        probas = F.softmax(logits, dim=1)
        return probas, hidden
    
    def init_hidden(self, batch_size:int=1):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.emb_size, device=device)
        return nn.init.xavier_uniform_(hidden_state)
        
class AttnDecoderLSTM(nn.Module):
    
    def __init__(self, emb_size:int, hidden_size:int, out_size:int, n_layers:int, dropout:float,
                 attention_version:str, max_length:bool=None):
        super(AttnDecoderLSTM, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.out_size = out_size # |V|
        self.n_layers = n_layers
        self.dropout = dropout
        self.attention_version = attention_version
        self.max_length = max_length # max source sequence length
        
        self.embedding = nn.Embedding(out_size, emb_size, padding_idx=0)
        
        if attention_version == 'general':
            isinstance(max_length, int), 'General attention requires maximum command sequence length'
            self.attention = GeneralAttention(emb_size, hidden_size, max_length)
            self.linear = nn.Linear(hidden_size, out_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        elif attention_version == 'multiplicative':
            self.attention = MultiplicativeAttention(hidden_size)
            self.linear = nn.Linear(hidden_size * 2, out_size)
            self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError("Attention version must be one of ['general', 'multiplicative']")

        
    def forward(self, x_batch, hidden, encoder_hiddens):
        batch_size = x_batch.size(0)
        # NOTE: first dim represents batch size, second represents sequence length, third dim embedding size (if batch_first=True)
        embedded = self.embedding(x_batch).view(batch_size, 1, -1)
        embedded = F.relu(embedded)
        
        if self.attention_version == 'multiplicative':            
            out, hidden = self.lstm(embedded, hidden)
            out = F.relu(out)
            context, attn_weights = self.attention(hidden[0][-1], encoder_hiddens)
            logits = self.linear(torch.cat((out, context), 2).squeeze(1))
        elif self.attention_version == 'general':
            context, attn_weights = self.attention(embedded, hidden[0][0], encoder_hiddens)
            out = F.relu(context)
            out, hidden = self.lstm(out, hidden)
            logits = self.linear(out.squeeze(1))
            
        probas = F.softmax(logits, dim=1)
        return probas, hidden # attn_weights
    
    def init_hidden(self, batch_size:int=1):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.emb_size, device=device)
        cell_state = torch.zeros(self.n_layers, batch_size, self.emb_size, device=device)
        hidden = (nn.init.xavier_uniform_(hidden_state), nn.init.xavier_uniform_(cell_state))
        return hidden
    
    
class AttnDecoderGRU(nn.Module):
    
    def __init__(self, emb_size:int, hidden_size:int, out_size:int, n_layers:int, dropout:float, 
                 attention_version:str, max_length:bool=None):
        super(AttnDecoderGRU, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.out_size = out_size # |V|
        self.n_layers = n_layers
        self.dropout = dropout
        self.attention_version = attention_version
        self.max_length = max_length # max target sequence length
        
        self.embedding = nn.Embedding(out_size, emb_size, padding_idx=0)
        
        if attention_version == 'general':
            isinstance(max_length, int), 'General attention requires maximum command sequence length'
            self.attention = GeneralAttention(emb_size, hidden_size, max_length)
            self.linear = nn.Linear(hidden_size, out_size)
            self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        elif attention_version == 'multiplicative':
            self.attention = MultiplicativeAttention(hidden_size)
            self.linear = nn.Linear(hidden_size * 2, out_size)
            self.gru = nn.GRU(emb_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError("Attention version must be one of ['general', 'multiplicative']")
            
       
    def forward(self, x_batch, hidden, encoder_hiddens):
        batch_size = x_batch.size(0)
        # NOTE: first dim represents batch size, second represents sequence length, third dim embedding size (if batch_first=True)
        embedded = self.embedding(x_batch).view(batch_size, 1, -1)
        embedded = F.relu(embedded)
        
        if self.attention_version == 'multiplicative':            
            out, hidden = self.gru(embedded, hidden)
            out = F.relu(out)
            context, attn_weights = self.attention(hidden[-1], encoder_hiddens)
            logits = self.linear(torch.cat((out, context), 2).squeeze(1))
        elif self.attention_version == 'general':
            context, attn_weights = self.attention(embedded, hidden[0], encoder_hiddens)
            out = F.relu(context)
            out, hidden = self.gru(out, hidden)
            logits = self.linear(out.squeeze(1))
        
        probas = F.softmax(logits, dim=1)
        return probas, hidden  # attn_weights (important only for visualisation)
    
    def init_hidden(self, batch_size:int=1):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.emb_size, device=device)
        return nn.init.xavier_uniform_(hidden_state)