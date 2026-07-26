
#this performs upsampling of the latent representation vector to the original audio input

import torch
import torch.nn as nn
import torch.nn.functional as F

from convolution.convblocks import CausalConv,CausalConvTranspose,CausalResBlock

class Decoder(nn.Module):
    #(B,D,T') -> (B,1,T)
    
    def __init__(
        self,
        out_channels: int=1,
        channels: int=32,
        latent_dim: int=128,
        strides: list[int] = [8,5,4,2], #reverse
        dilations: list[int] = [9,3,1], #for the 3 layers of residual blocks
    ):
        super().__init__()
        
        #initial layer: (remains the same as encoder)
        layers = [(
            CausalConv(
                latent_dim,
                out_channels,
                kernel_size=7
            )
        )]
        
        bottleneck_channels = channels * (2**(len(strides))) #?
        curr_channels = bottleneck_channels
        
        for s in strides:
            
            k = 2*s
            out_ch = curr_channels // 2 #flooring for upsampling
            
            for d in dilations: #3 residual layers
                
                layers.append(
                    CausalResBlock(
                        in_dim=curr_channels,
                        out_dim=out_ch, 
                        dilation=d
                    )
                )
            
            #introducing causal layers:
            layers.append(
                CausalConvTranspose(
                    input_dim=curr_channels,
                    output_dim=out_ch,
                    kernel_size=k,
                    stride=s
                )
            )
            layers.append(nn.ELU()) #why here? weird
            
            curr_channels = out_ch #updated from bottleneck_channels
            
        layers.append(
            CausalConv(
                in_dim = curr_channels,
                out_dim= out_channels,
                kernel_size=7
            )
        )
        
        #model:
        self.model = nn.Sequential(*layers)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        return self.model(x)
        
        
        
            
            
        
        