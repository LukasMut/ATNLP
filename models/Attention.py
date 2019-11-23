__all__ = ['GeneralAttention']

import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralAttention(nn.Module):
    """General attention version as introduced in Bahdanau et al. (2015) (look at https://arxiv.org/abs/1409.0473)
    """
    
    def __init__(self, hidden_size:int, seq_length:int):
        super(GeneralAttention, self).__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length # length of current source sentence (i.e., len(encoder_hiddens))

        self.attn = nn.Linear(self.hidden_size * 2, seq_length)
        self.attn_out = nn.Linear(self.hidden_size * 2, self.hidden_size)
                 
    def forward(self, embedded, hidden, encoder_hiddens):
        # squeeze removes dimension at specified position (0) --> 2D matrix (required for linear layer)
        attn_weights = F.softmax(self.attn(torch.cat((embedded.squeeze(0), hidden.squeeze(0)), 1)), dim=1) 
        # unsqueeze inserts dimension at specified position (0) --> 3D tensor (required for batch-matmul)
        context = torch.bmm(attn_weights.unsqueeze(0), encoder_hiddens.unsqueeze(0))
        # squeeze removes dimension at specified position (0) --> 2D matrix (required for linear layer)
        out = torch.cat((embedded.squeeze(0), context.squeeze(0)), 1)
        # unsqueeze inserts dimension at specified position (0) --> 3D tensor (required for RNN)
        out = self.attn_out(out).unsqueeze(0)
        return out, attn_weights
                           
                           