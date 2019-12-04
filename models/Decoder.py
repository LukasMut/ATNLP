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
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
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
    
    def __init__(self, emb_size:int, hidden_size:int, out_size:int, n_layers:int=2, dropout:float=0.5):
        super(DecoderGRU, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.out_size = out_size # |V|
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(out_size, emb_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
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
        cell_state = torch.zeros(self.n_layers, batch_size, self.emb_size, device=device)
        hidden = (nn.init.xavier_uniform_(hidden_state), nn.init.xavier_uniform_(cell_state))
        return hidden
    
    
class AttnDecoderRNN(nn.Module):
    
    def __init__(self, emb_size:int,  hidden_size:int, out_size:int, seq_length:int, n_layers:int=2, dropout_p:float=0.5):
        super(AttnDecoderRNN, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.out_size = out_size # |V|
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.seq_length = seq_length # length of the source sentence (i.e., len(encoder_hiddens))
        
        self.embedding = nn.Embedding(out_size, emb_size, padding_idx=0)
        self.attention = GeneralAttention(hidden_size, seq_length)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.linear = nn.Linear(hidden_size, out_size)
        
    def forward(self, x_batch, hidden, encoder_hiddens):
        batch_size = x_batch.size(0)
        # NOTE: first dim represents batch size, second represents sequence length, third dim embedding size (if batch_first=True)
        embedded = self.embedding(x_batch).view(batch_size, 1, -1)
        embedded = F.relu(embedded)
        out, attn_weights = self.attention(embedded, hidden, encoder_hiddens)
        out = F.relu(out)
        out, hidden = self.rnn(out, hidden)
        log_probas = F.log_softmax(self.linear(out.squeeze(0)), dim=1)
        return log_probas, hidden, attn_weights
    
    def init_hidden(self, batch_size:int=1):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.emb_size, device=device)
        return nn.init.xavier_uniform_(hidden_state)
        
class AttnDecoderLSTM(nn.Module):
    
    def __init__(self, emb_size:int, hidden_size:int, out_size:int, seq_length:int, n_layers:int=2, dropout_p:float=0.5):
        super(AttnDecoderLSTM, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.out_size = out_size # |V|
        self.n_layers = n_layers
        self.dropout = dropout
        self.seq_length = seq_length # length of the source sentence (i.e., len(encoder_hiddens))
        
        self.embedding = nn.Embedding(out_size, emb_size, padding_idx=0)
        self.attention = GeneralAttention(hidden_size, seq_length)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=False, dropout=dropout_p)
        self.linear = nn.Linear(hidden_size, out_size)
        
    def forward(self, x_batch, hidden, encoder_hiddens):
        batch_size = x_batch.size(0)
        # NOTE: first dim represents batch size, second represents sequence length, third dim embedding size (if batch_first=True)
        embedded = self.embedding(x_batch).view(batch_size, 1, -1)
        embedded = F.relu(embedded)
        out, attn_weights = self.attention(embedded, hidden, encoder_hiddens)
        out = F.relu(out)
        out, hidden = self.lstm(out, hidden)
        log_probas = F.log_softmax(self.linear(out.squeeze(0)), dim=1)
        return log_probas, hidden, attn_weights
    
    def init_hidden(self, batch_size:int=1):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.emb_size, device=device)
        cell_state = torch.zeros(self.n_layers, batch_size, self.emb_size, device=device)
        hidden = (nn.init.xavier_uniform_(hidden_state), nn.init.xavier_uniform_(cell_state))
        return hidden
    
    
class AttnDecoderGRU(nn.Module):
    
    def __init__(self, emb_size:int, hidden_size:int, out_size:int, seq_length:int, n_layers:int=2, dropout_p:float=0.5):
        super(AttnDecoderGRU, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.out_size = out_size # |V|
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.seq_length = seq_length # length of the source sentence (i.e., len(encoder_hiddens))
        
        self.embedding = nn.Embedding(out_size, emb_size, padding_idx=0)
        self.attention = GeneralAttention(hidden_size, seq_length)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.linear = nn.Linear(hidden_size, out_size)
        
    def forward(self, x_batch, hidden, encoder_hiddens):
        batch_size = x_batch.size(0)
        # NOTE: first dim represents batch size, second represents sequence length, third dim embedding size (if batch_first=True)
        embedded = self.embedding(x_batch).view(batch_size, 1, -1)
        embedded = F.relu(embedded)
        out, attn_weights = self.attention(embedded, hidden, encoder_hiddens)
        out = F.relu(out)
        out, hidden = self.gru(out, hidden)
        log_probas = F.log_softmax(self.linear(out.squeeze(0)), dim=1) # transform 3D tensor into 2D matrix for linear layer
        return log_probas, hidden, attn_weights
    
    def init_hidden(self, batch_size:int=1):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.emb_size, device=device)
        cell_state = torch.zeros(self.n_layers, batch_size, self.emb_size, device=device)
        hidden = (nn.init.xavier_uniform_(hidden_state), nn.init.xavier_uniform_(cell_state))
        return hidden
