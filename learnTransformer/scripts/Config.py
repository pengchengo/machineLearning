# encoding:utf-8

import torch

batch_size = 128
d_model = 512
nhead = 8
num_layers = 6
num_heads = 8
dropout = 0.1
max_len = 100
total_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 词表特殊符号与固定索引（与 Embedding padding_idx 等需一致）
PAD = "<pad>"
UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
tgt_vocab_size = 0
src_vocab_size = 0

init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 1000
clip = 1.0
weight_decay = 5e-4
inf = float('inf')
