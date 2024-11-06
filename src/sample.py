import torch
class Config:
 def __init__(self):
    self.device = 'cpu'  # 'cuda' 'cpu' 
    self.dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'


# Model