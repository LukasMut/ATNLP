import torch
import torch.nn as nn

# simple Elman RNN
class VanillaRNN(nn.Module):
    def __init__(self, in_size, hidden_size, n_layers:int=1, dropout:float=0.5, bidir:bool=False):
        super(EncoderRNN, self).__init__()
        
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidir = bidir
        
        self.embedding = nn.Embedding(in_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, batch_first=True, droput=dropout, bidirectional=bidir)
        
    def forward(self, word_inputs, hidden):
        # NOTE: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        # NOTE: first dim must represent batch size, second dim sequence length, third dim embedding size (if batch_first=True)
        embedded = self.embedding(word_inputs).view(1, seq_len, -1)
        out, hidden = self.rnn(embedded, hidden)
        return out, hidden

    def init_hidden(self):
        #TODO: fill this function