# encoding:utf-8

import torch
import torch.nn as nn

from learnTransformer.scripts.MutiHeadAttention import MutiHeadAttention
from learnTransformer.scripts.LayerNorm import LayerNorm
from learnTransformer.scripts.PositionwiseFeedForward import PositionwiseFeedForward
from learnTransformer.scripts.Config import dropout

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = MutiHeadAttention()
        self.norm1 = LayerNorm()
        self.dropout1 = nn.Dropout(dropout)
        self.cross_attention = MutiHeadAttention()
        self.norm2 = LayerNorm()
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward()
        self.norm2 = LayerNorm()
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        _x = x
        x = self.self_attention(x, x, x, tgt_mask)

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if encoder_output is not None:
            _x = x
            x = self.cross_attention(x, encoder_output, encoder_output, src_mask)

            x = self.dropout2(x)
            x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
