# encoding:utf-8

import torch
import torch.nn as nn
from learnTransformer.scripts.DecoderLayer import DecoderLayer
from learnTransformer.scripts.Embedding import Embedding
from learnTransformer.scripts.Config import num_layers, d_model
from learnTransformer.scripts import Config
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding(Config.tgt_vocab_size)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, Config.tgt_vocab_size)

    def forward(self, tgt, encoder_output, src_mask, tgt_mask):
        x = self.embedding(tgt)

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        output = self.linear(x)
        return output
        
        