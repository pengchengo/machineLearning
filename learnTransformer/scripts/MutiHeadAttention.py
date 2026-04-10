# encoding:utf-8

import torch
import torch.nn as nn

from learnTransformer.scripts.Config import d_model, nhead
from learnTransformer.scripts.ScaleDotProductAttention import ScaleDotProductAttention

class MutiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        q, k, v = self.w_q(Q), self.w_k(K), self.w_v(V)
        q, k, v = self.split(q), self.split(k), self.split(v)

        out, attention = self.attention(q, k, v, mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out

    def split(self, x):
        batch_size, length, d_model = x.size()
        d_tensor = d_model // nhead
        x = x.view(batch_size, length, nhead, d_tensor)
        return x

    def concat(self, x):
        batch_size, length, nhead, d_tensor = x.size()
        d_model = nhead * d_tensor
        x = x.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return x