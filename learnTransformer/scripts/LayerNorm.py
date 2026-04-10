# encoding:utf-8

import torch
import torch.nn as nn

from learnTransformer.scripts.Config import d_model

class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = 1e-6

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        x = self.gamma * x + self.beta
        return x
