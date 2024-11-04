import torch
import sys
from pathlib import Path
import time
from dataset import x0dataset

class Config:
    def __init__(self):
        self.log_dir = '/'
        self.checkpoint_dir = '/'
        self.data_dir = '/'
        self.disable_progress_bar = False
        self.n_vocab = 0
        self.n_ctx = 1024
        self.n_embd = 768
        self.n_head = 12
        self.n_layer = 12
        self.bench_size = 12
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'