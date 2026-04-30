# encoding:utf-8

import torch
import torch.nn as nn
from learnTransformer.scripts.EncoderLayer import EncoderLayer
from learnTransformer.scripts.Embedding import Embedding
from learnTransformer.scripts.Config import num_layers
from learnTransformer.scripts import Config

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding(Config.src_vocab_size)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(num_layers)])

    def forward(self, src, src_mask):
        x = self.embedding(src)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x