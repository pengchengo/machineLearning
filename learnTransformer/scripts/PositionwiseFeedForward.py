# encoding:utf-8

import torch
import torch.nn as nn

from learnTransformer.scripts.Config import d_model, dropout

class PositionwiseFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
