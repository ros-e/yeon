from dataclasses import dataclass
import torch 
import torch.nn
import numpy as npy

@dataclass
class LLMConfig:
    block_size: int
    n_vocab: int = 0
    n_ctx: int = 1024 
    # 124M params
    n_embd: int = 768
    n_head: int = 1
    n_layer: int = 12

def generate(self, temperature=1.0, top_k=None):
    pass