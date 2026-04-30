# encoding:utf-8

import torch
import torch.nn as nn
from learnTransformer.scripts.TokenEmbedding import TokenEmbedding
from learnTransformer.scripts.PositionalEncoding import PositionalEncoding
from learnTransformer.scripts.Config import dropout

class Embedding(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size)
        self.position_embedding = PositionalEncoding()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(x)
        return self.dropout(tok_emb + pos_emb)