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

class Trainer:
    def __init__(self):
        self.parse_args()
        self.train()
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--log-dir', type=str, required=True, help='Directory to store training logs')
        parser.add_argument('--checkpoint-dir', type=str, required=True, help='Directory to store model checkpoints')
        parser.add_argument('--data-dir', type=str, required=True, help='Directory containing training data')
        parser.add_argument('--disable-progress-bar', action='store_true', help='Disables progress bar')
        
        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)
        self.args = parser.parse_args()

    def train(self):
        if self.args.disable_progress_bar:
            print("Progress bar is disabled.")
        else:
            print("Progress bar is enabled.")

if __name__ == '__main__':
    trainer = Trainer()
    args = trainer.parse_args()