# encoding:utf-8

from tabnanny import verbose
import time

import torch
from learnTransformer.scripts.Transformer import Transformer
from learnTransformer.scripts.Config import device, batch_size, total_epochs
import torch.optim as optim
from learnTransformer.scripts.Config import init_lr, factor, adam_eps, patience, warmup, epoch, clip, weight_decay, inf
import torch.nn as nn
from learnTransformer.scripts.DataLoader import get_dataloaders

model = Transformer()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=init_lr, eps=adam_eps, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor=factor, patience=patience)
criterion = nn.CrossEntropyLoss()

train_loader, val_loader, src_vocab, tgt_vocab = get_dataloaders(batch_size=batch_size)
    

def train():
    model.train()
    epoch_loss = 0.0
    for src, trg in train_loader:
        # src: [batch, src_len], tgt: [batch, tgt_len]，含 <sos>...<eos>
        optimizer.zero_grad()
        output = model(src, trg[:,:-1])
        output_reshape = output.contiguous().view(-1, output.shape(-1))
        trg = trg[:,1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

    return epoch_loss / len(train_loader)
        
def evaluate():
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for src, trg in val_loader:
            output = model(src, trg[:,:-1])
            output_reshape = output.contiguous().view(-1, output.shape(-1))
            trg = trg[:,1:].contiguous().view(-1)
            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(val_loader)

def run():
    best_loss = float('inf')
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
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(best_loss))
        print(f"Val Loss: {val_loss}")

if __name__ == "__main__":
    run()