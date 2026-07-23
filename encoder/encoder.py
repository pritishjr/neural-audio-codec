
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    
    def __init__(self, conv_strides: list[int], input_dim: 1, latent_dim: 128 ):
        super().__init__()
        
        self.conv_strides = [2,4,5,8] 
        
        