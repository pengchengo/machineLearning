# encoding:utf-8

import torch
import torch.nn as nn

from learnTransformer.scripts.MutiHeadAttention import MutiHeadAttention
from learnTransformer.scripts.LayerNorm import LayerNorm
from learnTransformer.scripts.PositionwiseFeedForward import PositionwiseFeedForward
from learnTransformer.scripts.Config import dropout

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = MutiHeadAttention()
        self.norm1 = LayerNorm()
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward()
        self.norm2 = LayerNorm()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        _x = x
        x = self.attntion(x, x, x, mask)

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
