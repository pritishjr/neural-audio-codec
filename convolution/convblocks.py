
#This we are using to make the convolutional blocks that help the raw audio chunks to be converted into its latent representations.
#thus, this helps us give the causality to the input stream.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

class CausalConv(nn.Module):
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stride: int = 1,
        kernel_size: int = 1,
        dilation: int = 1,
        groups: int = 1 #?
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        #for making it causal: we apply a padding
        self.causal_padding = (kernel_size - 1)*(dilation)        
        
        #we have to apply this padding to the input dimension:
        self.conv = weight_norm(
            nn.Conv1d(
                in_dim,
                out_dim,
                kernel_size,
                stride,
                padding=0, #padding is handled explicitly during forward pass.
                groups=groups
            )
        )
        
    def forward(self, x: torch.Tensor) -> torch.tensor: #this outputs a causal latent vector representation of the input stream of audio
        #padding: (left = (k-1)*d, right = 0)
        x = F.pad(x, (self.causal_padding,0))
        
        return self.conv(x)
    
class CausalConvTranspose(nn.Module):
    #to 
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.right_pad = kernel_size-stride
        
        self.conv_transpose = weight_norm(
            nn.ConvTranspose1d(
                in_channels = input_dim,
                out_channels=output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=0, #called expllicity
                bias=True
            )
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        #removing the future overhang on the right
        if self.right_pad > 0:
            x = x[:, :, :-self.right_pad]
        
        return x
        
class CausalResBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dilation: int
    ):
        super().__init__()
        
        
        self.res_block = weight_norm(
            nn.Sequential(
                nn.ELU(),
                CausalConv(
                    in_dim,
                    out_dim,
                    stride = 1,
                    kernel_size= 3,
                    dilation=dilation
                ),
                nn.ELU(),
                CausalConv(
                    in_dim,
                    out_dim,
                    kernel_size=1
                )
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        y = x + self.res_block(x)
        
        return y
    
        