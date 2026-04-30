# encoding:utf-8

from tabnanny import verbose
import time
import sys
from pathlib import Path

# 允许直接运行本文件时仍可解析 `learnTransformer.*` 绝对导入
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from learnTransformer.scripts import Config
from learnTransformer.scripts.Transformer import Transformer
from learnTransformer.scripts.Config import device, batch_size, total_epochs
import torch.optim as optim
from learnTransformer.scripts.Config import init_lr, factor, adam_eps, patience, warmup, epoch, clip, weight_decay, inf
import torch.nn as nn
from learnTransformer.scripts.DataLoader import get_dataloaders

train_loader, val_loader, src_vocab, tgt_vocab = get_dataloaders(batch_size=batch_size)
Config.src_vocab_size = len(src_vocab)
Config.tgt_vocab_size = len(tgt_vocab)

model = Transformer()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=init_lr, eps=adam_eps, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)
criterion = nn.CrossEntropyLoss()
    

def train():
    model.train()
    epoch_loss = 0.0
    for src, trg in train_loader:
        # src: [batch, src_len], tgt: [batch, tgt_len]，含 <sos>...<eos>
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg[:,:-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:,1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        print(f"Loss: {loss.item()}")

    return epoch_loss / len(train_loader)
        
def evaluate():
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for src, trg in val_loader:
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg[:,:-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:,1:].contiguous().view(-1)
            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(val_loader)

def run():
    best_loss = float('inf')
    save_dir = PROJECT_ROOT / "learnTransformer/saved"
    save_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(total_epochs):
        start_time = time.time()
        train_loss = train()
        val_loss = evaluate()
        end_time = time.time()
        print(f"Time: {end_time - start_time}s")
        scheduler.step(val_loss)
        print(f"Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), str(save_dir / f"model-{best_loss}.pt"))
        print(f"Val Loss: {val_loss}")

if __name__ == "__main__":
    run()