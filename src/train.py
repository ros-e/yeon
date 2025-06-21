import torch
import torch.nn as nn
from model import Model, ModelArgs
import requests
from tqdm import tqdm
import logging
import os
class CharDataset:
    def __init__(self, text, seq_length=100):
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.seq_length = seq_length
        self.text = text

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        chunk = self.text[idx:idx + self.seq_length + 1]
        input_seq = [self.char_to_idx[ch] for ch in chunk[:-1]]
        target_seq = [self.char_to_idx[ch] for ch in chunk[1:]]
        input_tensor = torch.zeros(self.seq_length, self.vocab_size)
        for i, char_idx in enumerate(input_seq):
            input_tensor[i, char_idx] = 1.0
        return input_tensor, torch.tensor(target_seq, dtype=torch.long)
def download_data(url: str, filename: str):
    if not os.path.exists(filename):
        logging.info(f"Downloading {filename}")
        response = requests.get(url)
        with open(filename, 'w', encoding='utf-8') as f:
                f.write(response.text)
        logging.info(f"Data saved to {filename}")
    else:
        logging.info(f"Data already exists at {filename}")
def create_dataloader(dataset, batch_size=32, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
def train_model(model, dataloader, loss_fn, optimizer, device, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, ncols=100)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets[:, -1]
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    DATAFILE = "shakespeare.txt"
    DATAURL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    LEARNINGRATE = 0.001
    logging.basicConfig(level=logging.INFO)
    os.makedirs("checkpoints", exist_ok=True)
    download_data(DATAURL, DATAFILE)
    with open(DATAFILE, 'r', encoding='utf-8') as f:
        text = f.read()
    logging.info(f"Text length: {len(text)} characters")
    dataset = CharDataset(text, seq_length=100)
    logging.info(f"Vocabulary size: {dataset.vocab_size}")
    logging.info(f"Characters: {''.join(dataset.chars[:50])}...")
    dataloader = create_dataloader(dataset, batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = ModelArgs()
    args.input_size = dataset.vocab_size
    args.output_size = dataset.vocab_size
    model = Model(args).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNINGRATE)
    train_model(model, dataloader, loss_fn, optimizer, device)
    #Save model & vocab
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': dataset.vocab_size,
        'char_to_idx': dataset.char_to_idx,
        'idx_to_char': dataset.idx_to_char,
        'model_args': args
    }, "checkpoints/model_checkpoint.pth")

if __name__ == "__main__":
    main()
