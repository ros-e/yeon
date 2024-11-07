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
def init(self, config: LLMConfig):
    super().init()
    self.config = config
    self.token_embedding = torch.nn.Embedding(config.n_vocab, config.n_embd)
    self.position_embedding = torch.nn.Embedding(config.block_size, config.n_embd)

def generate(self, temperature=1.0, top_k=None):
    pass