"""
python3.1 \
    --log-dir logs/
    --checkout-dir checpoint/
    --data-dir data/
    --disable-progress-bar
"""
import torch
import argparse
import sys
from pathlib import Path
import time
from dataset import data_loader

class Trainer:
    def __init__(self):
    #stolen from gpt2
        n_vocab = 0,
        n_ctx = 1024,
        n_embd = 768,
        n_head = 12,
        n_layer = 12,
        self.parse_args()
        self.train()
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', type=int, default=12, required=False, help='Number of samples processed in one iteration during training')
        parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='Specify the device to use for training: "cpu" or "cuda".')
        parser.add_argument('--log-dir', type=str, required=True, help='Directory to store training logs')
        parser.add_argument('--checkpoint-dir', type=str, required=True, help='Directory to store model checkpoints')
        parser.add_argument('--data-dir', type=str, required=True, help='Directory containing training data')
        parser.add_argument('--disable-progress-bar', action='store_true', help='Disables progress bar')
        
        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)
        self.args = parser.parse_args()

    def train(self):
        data_loader()
            