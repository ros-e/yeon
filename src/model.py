from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as num

@dataclass
class ModelArgs:
    """
    max_batch_size
    vocab_size
    device | cpu, cuda
    """
    max_batch_size: int = 32
    vocab_size: int = 0
    device: str = 'cpu'

class RNN():


class Model(nn.Module):
    def __init__(self, modelargs: ModelArgs):
        super().__init__()
        self.args = modelargs

    def foward(self, x: torch.Tensor) -> torch.Tensor:
        
        pass

