# encoding:utf-8

import torch
import torch.nn as nn
from learnTransformer.scripts.Encoder import Encoder
from learnTransformer.scripts.Decoder import Decoder
from learnTransformer.scripts.Config import PAD_IDX

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src = self.encoder(src, src_mask)
        tgt = self.decoder(tgt, src, src_mask, tgt_mask)
        return tgt

    def make_src_mask(self, src):
        src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        tgt_mask = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)
        return tgt_mask
