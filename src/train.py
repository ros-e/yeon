"""
python3.11 src/train.py \
  --log_dir "./logs" \
  --checkpoint_dir "./checkpoints" \
  --device "mps" \
  --bench_size 164
"""
import os
import requests
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration for language model.")
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to store dataset')
    parser.add_argument('--n_vocab', type=int, default=0, help='Vocabulary size')
    parser.add_argument('--n_ctx', type=int, default=1024, help='Context length')
    parser.add_argument('--n_embd', type=int, default=768, help='Embedding size')
    parser.add_argument('--n_head', type=int, default=12, help='Number of heads in multihead attention')
    parser.add_argument('--n_layer', type=int, default=12, help='Number of layers')
    parser.add_argument('--bench_size', type=int, default=12, help='Batch size for training')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='Device to train on')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], 
                        help='Data type for training')
    
    args = parser.parse_args()
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    
    return args

def download_data(config):
    input_file_path = os.path.join(config.data_dir, 'input.txt')
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
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
    device = torch.device(config.device)
    train_data = torch.tensor(train_ids, dtype=torch.long, device=device)
    inputs = train_data[:-1]
    targets = train_data[1:]
    vocab_size = 256
    hidden_size = 128
    model = torch.nn.Sequential(
        torch.nn.Embedding(vocab_size, hidden_size),
        torch.nn.Linear(hidden_size, vocab_size)
    ).to(device)  
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
            input_char = torch.tensor([ord(start_char)], dtype=torch.long, device=device)
            for _ in range(100):
                with torch.no_grad():
                    output = model(input_char)
                    prob = torch.nn.functional.softmax(output[0], dim=0)
                    next_char_idx = torch.multinomial(prob, 1).item()
                    next_char = chr(next_char_idx)
                    generated += next_char
                    input_char = torch.tensor([next_char_idx], dtype=torch.long, device=device)
    checkpoint_path = str(Path(config.checkpoint_dir) / 'shakespeare_model.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved: {checkpoint_path}")
if __name__ == "__main__":
    config = parse_args()
    train(config)
