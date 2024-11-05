from dataclasses import dataclass
import torch 
import numpy as mpy

def load_pretrained():
    pass

@dataclass
class LLMConfig:
    block_size: int
    n_vocab: int = 0
    n_ctx: int = 1024 
    # 124M params
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12

class Model(torch.nn):
    def __init__(self, config):
        super().__init__()
