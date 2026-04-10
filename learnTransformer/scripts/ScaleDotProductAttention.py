# encoding:utf-8

import torch
import torch.nn as nn
from learnTransformer.scripts.Config import max_len, d_model, device
import math

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = k.size()

        k_t = k.transpose(2, 3)
        score = (q @ k_t) /math.sqrt(d_tensor)

        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        score = self.softmax(score)
        output = score @ v
        return v,score
