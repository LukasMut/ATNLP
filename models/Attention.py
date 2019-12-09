__all__ = ['GeneralAttention', 'MultiplicativeAttention']

import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralAttention(nn.Module):
    """
    General attention version as introduced in Bahdanau et al. (2015) (https://arxiv.org/abs/1409.0473)
    """
    
    def __init__(self, hidden_size:int, max_length):
        super(GeneralAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_out = nn.Linear(self.hidden_size * 2, self.hidden_size)
                 
    def forward(self, embedded, hidden, encoder_outputs):        
        # squeeze removes dimension at specified position (0) --> 2D matrix (required for linear layer)
        attn_scores = self.attn(torch.cat((embedded.squeeze(0), hidden.squeeze(0)), 1))
        attn_weights = F.softmax(attn_scores, dim=1)
        # unsqueeze inserts dimension at specified position (0) --> 3D tensor (required for batch-matmul)
        context = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # squeeze removes dimension at specified position (0) --> 2D matrix (required for linear layer)
        out = torch.cat((embedded.squeeze(0), context.squeeze(0)), 1)
        # unsqueeze inserts dimension at specified position (0) --> 3D tensor (required for RNN)
        out = self.attn_out(out).unsqueeze(0)
        return out, attn_weights
    
    
class MultiplicativeAttention(nn.Module):
    """
    Multiplicative attention version as introduced in Luong et al. (2015) (https://arxiv.org/pdf/1508.04025.pdf)
    """
    def __init__(self, hidden_size:int):
        super(MultiplicativeAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden, encoder_outputs):
        key = self.attn(encoder_outputs)
        attn_scores = torch.sum(hidden.unsqueeze(1) * key, dim=2) # broadcasting, sum along the sequence [32, 10]
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(1)
        context = attn_weights.bmm(encoder_outputs) # [32, 1, 10] x [32, 10, 500]
        return context, attn_weigths