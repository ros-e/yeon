from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class ModelArgs:
    """
    input_size: int
    hidden_size: int
    output_size: int
    """
    input_size: int = 64
    hidden_size: int = 128
    output_size: int = 1000

class Model(nn.Module):
    def __init__(self, modelargs: ModelArgs):
        super().__init__()
        self.hidden_size = modelargs.hidden_size
        self.rnn = nn.RNN(
            input_size=modelargs.input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.linear = nn.Linear(self.hidden_size * 2, modelargs.output_size)
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(2 * 2, batch_size, self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        return self.linear(out)
