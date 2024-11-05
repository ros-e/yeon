import os
import requests
import tiktoken
import numpy as np
import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import time

class Config:
    def __init__(self):
        self.log_dir = './logs'
        self.checkpoint_dir = './checkpoints'
        self.data_dir = './data'
        self.n_vocab = 0
        self.n_ctx = 1024
        self.n_embd = 768
        self.n_head = 12
        self.n_layer = 12
        self.bench_size = 12
        self.device = 'cpu'  # 'cuda' 'cpu'
        self.dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

def download_data(config):
    # Download the tiny Shakespeare dataset if not present
    input_file_path = os.path.join(config.data_dir, 'input.txt')
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt' #replace this with a playboi carti dataset later
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)
    return input_file_path

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]
    train_ids = [ord(c) for c in train_data]
    val_ids = [ord(c) for c in val_data]
    print(f"Train has {len(train_ids):,} characters")
    print(f"Validation has {len(val_ids):,} characters")
    np.array(train_ids, dtype=np.uint16).tofile(os.path.join(os.path.dirname(file_path), 'train.bin'))
    np.array(val_ids, dtype=np.uint16).tofile(os.path.join(os.path.dirname(file_path), 'val.bin'))
    return train_ids, val_ids

def train(config):
    input_file_path = download_data(config)
    train_ids, val_ids = load_data(input_file_path)
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(10):
        for inputs, targets in DataLoader([(torch.rand(10), torch.rand(1)) for _ in range(100)], 
           batch_size=config.bench_size, shuffle=True):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.MSELoss()(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")
            checkpoint_path = str(Path(config.checkpoint_dir) / f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    conf = Config()
    train(conf)
