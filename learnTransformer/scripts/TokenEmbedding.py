# encoding:utf-8

import torch
import torch.nn as nn
from learnTransformer.scripts.Config import d_model, PAD_IDX

class TokenEmbedding(nn.Embedding):
    def __init__(self):
        super().__init__(vocab_size, d_model, padding_idx=PAD_IDX)
