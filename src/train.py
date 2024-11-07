import os
import requests
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt' # Replace this with a playboi carti dataset later
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
    train_data = torch.tensor(train_ids, dtype=torch.long)
    inputs = train_data[:-1]
    targets = train_data[1:]
    vocab_size = 256
    hidden_size = 128
    model = torch.nn.Sequential(
        torch.nn.Embedding(vocab_size, hidden_size),
        torch.nn.Linear(hidden_size, vocab_size)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    batch_size = config.bench_size
    num_batches = (len(inputs) - batch_size) // batch_size
    for i in tqdm(range(0, len(inputs) - batch_size, batch_size), total=num_batches, desc="Training"):
        batch_inputs = inputs[i:i + batch_size]
        batch_targets = targets[i:i + batch_size]
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            start_char = 'T'
            generated = start_char
            input_char = torch.tensor([ord(start_char)], dtype=torch.long)
            for _ in range(100):
                with torch.no_grad():
                    output = model(input_char)
                    prob = torch.nn.functional.softmax(output[0], dim=0)
                    next_char_idx = torch.multinomial(prob, 1).item()
                    next_char = chr(next_char_idx)
                    generated += next_char
                    input_char = torch.tensor([next_char_idx], dtype=torch.long)
    checkpoint_path = str(Path(config.checkpoint_dir) / 'shakespeare_model.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved: {checkpoint_path}")

if __name__ == "__main__":
    conf = Config()
    train(conf)