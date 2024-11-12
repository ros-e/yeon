from dataclasses import dataclass
import torch 
import torch.nn
import numpy as np

@dataclass
class LLMConfig:
    block_size: int
    n_vocab: int = 0
    n_ctx: int = 1024 
    n_embd: int = 768
    n_head: int = 1
    n_layer: int = 12
    dropout: float = 0.1

class FeedForward(torch.nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(config.n_vocab, config.n_embd)
        self.position_embedding = torch.nn.Embedding(config.block_size, config.n_embd)
        self.blocks = torch.nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.lm_head = torch.nn.Linear(config.n_embd, config.n_vocab, bias=False)
        self.config = config

class TransformerBlock(torch.nn.Module): 
    def __init__(self, config: LLMConfig):
        super().__init__() 
        self.token_embedding = torch.nn.Embedding(config.n_vocab, config.n_embd)
        self.attn = torch.nn.MultiheadAttention(config.n_embd, config.n_head)
         # Feed Foward Network / FFN (haha funny goverment 3l sounding name)
        self.ffn = torch.nn.Sequential(
         torch.nn.Linear(config.n_embd, 4 * config.n_embd),
         torch.nn.ReLU(),
         torch.nn.Linear(4 * config.n_embd, config.n_embd),
         torch.nn.Dropout(config.dropout)
        )
        self.config = config

    def forward(self, x):
        b, t = x.size()
        token_emb = self.token_embedding(x)
        # TODO: add block stuff here 
        return x

    def generate(self, temperature=1.0, top_k=None):
        pass