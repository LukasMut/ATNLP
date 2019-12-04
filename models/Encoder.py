__all__ = ['EncoderRNN', 'EncoderLSTM', 'EncoderGRU']

import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# Vanilla RNN implementation
class EncoderRNN(nn.Module):
    def __init__(self, in_size:int, emb_size:int, hidden_size:int, n_layers:int=2, dropout:float=0.5, bidir:bool=False):
        super(EncoderRNN, self).__init__()
        
        self.in_size = in_size # |V|
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidir = bidir
        
        self.embedding = nn.Embedding(in_size, emb_size, padding_idx=0)
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, batch_first=True, dropout=dropout, bidirectional=bidir)
        
    def forward(self, x_batch, x_lengths, hidden):
        # NOTE: we run this all at once (over the whole input sequence)
        batch_size, seq_len = x_batch.shape
        # NOTE: first dim represents batch size, second represents sequence length, third dim embedding size (if batch_first=True)
        embedded = self.embedding(x_batch).view(batch_size, seq_len, -1)
        # move x_lengths to CPU 
        x_lengths = x_lengths.detach().cpu().numpy()
        packed = nn.utils.rnn.pack_padded_sequence(embedded, x_lengths, batch_first=True)
        out, hidden = self.rnn(packed, hidden)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # if bidirectional, sum outputs of LSTM
        if self.bidir:
            out = out[:, :, :self.hidden_size] + out[:, : ,self.hidden_size:]
        return out, hidden

    def init_hidden(self, batch_size:int=1):
        # NOTE: we need to initialise twice as many hidden states for bidirectional neural networks
        n = self.n_layers * 2 if self.bidir else self.n_layers
        hidden_state = torch.zeros(n, batch_size, self.hidden_size, device=device)
        return nn.init.xavier_uniform_(hidden_state)
    
    
# Vanilla LSTM implementation
class EncoderLSTM(nn.Module):
    def __init__(self, in_size:int, emb_size:int, hidden_size:int=200, n_layers:int=2, dropout:float=0.5, bidir:bool=False):
        super(EncoderLSTM, self).__init__()
        
        self.in_size = in_size # |V|
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidir = bidir
        
        self.embedding = nn.Embedding(in_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True, dropout=dropout, bidirectional=bidir)
        
    def forward(self, x_batch, x_lengths, hidden):
        # NOTE: we run this all at once (over the whole input sequence)
        batch_size, seq_len = x_batch.shape
        # NOTE: first dim represents batch size, second represents sequence length, third dim embedding size (if batch_first=True)
        embedded = self.embedding(x_batch).view(batch_size, seq_len, -1)
        # move x_lengths to CPU 
        x_lengths = x_lengths.detach().cpu().numpy()
        packed = nn.utils.rnn.pack_padded_sequence(embedded, x_lengths, batch_first=True)
        out, hidden = self.lstm(packed, hidden)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # if bidirectional, sum outputs of LSTM
        if self.bidir:
            out = out[:, :, :self.hidden_size] + out[:, : ,self.hidden_size:]
        return out, hidden
    
    def init_hidden(self, batch_size:int=1):
        # NOTE: we need to initialise twice as many hidden states for bidirectional neural networks
        n = self.n_layers * 2 if self.bidir else self.n_layers 
        hidden_state = torch.zeros(n, batch_size, self.hidden_size, device=device)
        cell_state = torch.zeros(n, batch_size, self.hidden_size, device=device)
        hidden = (nn.init.xavier_uniform_(hidden_state), nn.init.xavier_uniform_(cell_state))
        return hidden
    
# Vanilla GRU implementation
class EncoderGRU(nn.Module):
    def __init__(self, in_size:int, emb_size:int, hidden_size:int=200, n_layers:int=2, dropout:float=0.5, bidir:bool=False):
        super(EncoderGRU, self).__init__()
        
        self.in_size = in_size # |V|
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidir = bidir
        
        self.embedding = nn.Embedding(in_size, emb_size, padding_idx=0)
        self.gru = nn.GRU(emb_size, hidden_size, n_layers, batch_first=True, dropout=dropout, bidirectional=bidir)
        
    def forward(self, x_batch, x_lengths, hidden):
        # NOTE: we run this all at once (over the whole input sequence)
        batch_size, seq_len = x_batch.shape
        # NOTE: first dim represents batch size, second represents sequence length, third dim embedding size (if batch_first=True)
        embedded = self.embedding(x_batch).view(batch_size, seq_len, -1)
        # move x_lengths to CPU 
        x_lengths = x_lengths.detach().cpu().numpy()
        packed = nn.utils.rnn.pack_padded_sequence(embedded, x_lengths, batch_first=True)
        out, hidden = self.gru(packed, hidden)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # if bidirectional, sum outputs of LSTM
        if self.bidir:
            out = out[:, :, :self.hidden_size] + out[:, : ,self.hidden_size:]
        return out, hidden
    
    def init_hidden(self, batch_size:int):
        # NOTE: we need to initialise twice as many hidden states for bidirectional neural networks
        n = self.n_layers * 2 if self.bidir else self.n_layers 
        hidden_state = torch.zeros(n, batch_size, self.hidden_size, device=device)
        cell_state = torch.zeros(n, batch_size, self.hidden_size, device=device)
        hidden = (nn.init.xavier_uniform_(hidden_state), nn.init.xavier_uniform_(cell_state))
        return hidden