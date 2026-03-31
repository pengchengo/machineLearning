# encoding:utf-8

import torch
import torch.nn as nn
from learnTransformer.scripts.Encoder import Encoder
from learnTransformer.scripts.Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, src, tgt):
        src = self.encoder(src)
        tgt = self.decoder(tgt)
        return self.output(tgt)
