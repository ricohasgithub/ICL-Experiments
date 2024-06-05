
import math
from typing import Any, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    "Implements Standard MultiHeadAttention"

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

class CausalAttention(Attention):
    
    def __init__(self, n_heads=8, d_hidden=64, p_dropout=0.0, scaling=1.0, bias=True):
        super(CausalAttention, self).__init__(n_heads, d_hidden, p_dropout, scaling, bias)

    def forward(self, x, y=None, mask=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        t = torch.arange(seq_len)
        causal_mask = (t[:, None] >= t[None, :])[None, None, :, :]
        if mask is None:
            mask = torch.broadcast(causal_mask, (batch_size, 1, seq_len, seq_len))
        else:
            mask = mask * causal_mask
        return super(CausalAttention, self)(x=x, y=y, mask=mask)


class Dense(nn.Module):
    def __init__(self, in_features, widening_factor=4, p_dropout=0.1, init_scale=1.0):
        super(Dense, self).__init__()
        out_features = widening_factor * in_features
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, in_features)
        self.p_dropout = p_dropout
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.linear1.weight, a=0, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.linear1.bias)
        nn.init.kaiming_uniform_(self.linear2.weight, a=0, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.linear2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, causal=True, widening_factor=4, n_heads=8, d_hidden=64, p_dropout=0.1, scaling=1.0, bias=True):
        super(TransformerBlock, self).__init__()

        self.causal = causal
        self.widening_factor = widening_factor
        self.n_heads = n_heads
        self.d_hidden = d_hidden
        self.p_dropout = p_dropout
        self.scaling = scaling
        self.bias = bias

        self.layer_norm = LayerNorm(self.d_hidden)
        self.causal_block = CausalAttention(self.n_heads, self.d_hidden, self.p_dropout, self.scaling, self.bias)
        self.attention_block = Attention(self.n_heads, self.d_hidden, self.p_dropout, self.scaling, self.bias)
        self.dense_block = Dense(in_features=self.d_hidden, widening_factor=self.widening_factor, p_dropout=self.p_dropout)

    def forward(self, x, y=None, mask=None):
        if (self.causal):
            x += self.causal_block(x, y, mask)
        else:
            x += self.attention_block(x, y, mask)
        x += self.dense_block(x)
        return x

class Transformer(nn.Module):
    
    def __init__(self, input_embedder, n_classes=1623, n_layers=8, n_heads=8, p_dropout=0.1, d_hidden=64):
        super(Transformer, self).__init__()
        self.input_embedder = input_embedder
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.d_hidden = d_hidden

        self.layer_norm = LayerNorm(self.d_hidden)
        for i in range(self.n_layers):
            setattr(self, f'transformer_block_{i}', TransformerBlock(causal=True, widening_factor=4, n_heads=self.n_heads, d_hidden=self.d_hidden, p_dropout=self.p_dropout))
        self.linear = nn.Linear(self.d_hidden, self.n_classes)

        nn.init.kaiming_uniform_(self.linear.weight, a=0, mode='fan_in', nonlinearity='linear')


    def forward(self, examples, labels, mask=None, is_training=True):
        x = self.input_embedder(examples, labels, is_training)

        if mask is not None:
            attention_mask = mask[:, None, None, :]
        else:
            attention_mask = None

        for i in range(self.n_layers):
            if mask is not None:
                x *= mask[:, :, None]
            x = getattr(self, f'transformer_block_{i}')(x, mask=attention_mask)

        x = self.layer_norm(x)
        if mask is not None:
            x *= mask[:, :, None]
        
        return self.linear(x)
        
