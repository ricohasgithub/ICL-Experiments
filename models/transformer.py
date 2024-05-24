
import math

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    "Code from: https://nlp.seas.harvard.edu/annotated-transformer/"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Attention(nn.Module):

    def __init__(self, n_heads=8, d_hidden=64, p_dropout=0.0, scaling=1.0, bias=True):

        super(Attention, self).__init__()

        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.scaling = scaling
        self.bias = bias

        self.W_Q = nn.Linear(d_hidden, d_hidden)
        self.W_K = nn.Linear(d_hidden, d_hidden)
        self.W_V = nn.Linear(d_hidden, d_hidden)
        self.W_O = nn.Linear(d_hidden, d_hidden)

    def forward(self, x, y=None, mask=None):

        batch_size, seq_len = x.shape[0], x.shape[1]

        # If a mask is given, add an extra dimension so the same mask can be applied over all heads
        if mask is not None:
            mask = mask.unsqueeze(1)

        if y is None:
            y = x

        Q = self.W_Q(x).view(batch_size, -1, self.n_heads, self.d_hidden // self.n_heads)
        K = self.W_K(x).view(batch_size, -1, self.n_heads, self.d_hidden // self.n_heads)
        V = self.W_V(x).view(batch_size, -1, self.n_heads, self.d_hidden // self.n_heads)

        x, att_dist = self.attention(Q, K, V, mask)
        x = (x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_hidden))
        return self.W_O(x)

    def attention(self, Q, K, V, mask=None):
        d_K = Q.size(-1)
        att_scores = torch.matmul(Q, K.tranpose(-2, -1)) / math.sqrt(d_K)
        if mask is not None:
            att_scores = att_scores.masked_fill(mask == 0, -1e9)
        att_dist = att_scores.softmax(dim=-1)
        return torch.matmul(att_dist, V), att_dist

class Transformer(nn.Module):
    
    def __init__(self):
        pass
