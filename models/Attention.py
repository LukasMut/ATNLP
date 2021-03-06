__all__ = ['GeneralAttention', 'MultiplicativeAttention']

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class GeneralAttention(nn.Module):
    """
    General attention version as introduced in Bahdanau et al. (2015) (https://arxiv.org/abs/1409.0473)
    """
    
    def __init__(self, emb_size:int, hidden_size:int, max_length):
        super(GeneralAttention, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        self.attn = nn.Linear(self.emb_size + self.hidden_size, self.max_length)
        self.attn_out = nn.Linear(self.emb_size + self.hidden_size, self.hidden_size)
                 
    def forward(self, embedded, hidden, encoder_outputs):
        # squeeze removes dimension at specified position (0) --> 2D matrix (required for linear layer)
        attn_scores = self.attn(torch.cat((embedded.squeeze(1), hidden), 1))
        attn_weights = F.softmax(attn_scores, dim = 1).unsqueeze(1)
        # unsqueeze inserts dimension at specified position (0) --> 3D tensor (required for batch-matmul)
        context = attn_weights.bmm(encoder_outputs) # [32, 1, 10] x [32, 10, 100] batch-matmul
        # squeeze removes dimension at specified position (0) --> 2D matrix (required for linear layer)
        out = torch.cat((embedded.squeeze(1), context.squeeze(1)), 1)
        # unsqueeze inserts dimension at specified position (0) --> 3D tensor (required for RNN)
        out = self.attn_out(out).unsqueeze(1)
        return out, attn_weights
    

# TODO: implement scaled version of multiplicative attention (see "Attention is all you need" for further details)
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
        attn_weights = F.softmax(attn_scores, dim = 1).unsqueeze(1)
        context = attn_weights.bmm(encoder_outputs) # [32, 1, 10] x [32, 10, 500]
        return context, attn_weights