# encoding:utf-8

import torch
import torch.nn as nn
import learnTransformer.scripts.EncoderLayer as EncoderLayer
from learnTransformer.scripts.Embedding import Embedding
from learnTransformer.scripts.Config import num_layers

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(num_layers)])

    def forward(self, src, src_mask):
        x = self.embedding(src)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x