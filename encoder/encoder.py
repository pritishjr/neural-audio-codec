
#basically what this thing does is:
#converts: input - (B,1,T) to output - (B, D, T')
# T' = T/hop_size


import torch
import torch.nn as nn
import torch.nn.functional as F

from convolution.convblocks import CausalConv, CausalResBlock

class Encoder(nn.Module):
    
    def __init__(
        self,
        conv_strides: list[int] = [2,4,5,8], #stages of downsampling
        input_dim: int = 1,
        channels: int = 32, #feature-channels capturing the internal neural capacity.
        latent_dim: int = 128,
        dilations: int = [1,3,9] 
    ):
        super().__init__()
        
        # self.conv_strides = [2,4,5,8] #hop size = 320
        
        #keeping hop size as 320: 320 samples per audio chunk (when sampled at 24kHz - 13.3ms per chunk)
        
        self.hop_size = 1
        for i in self.conv_strides:
            self.hop_size *= i
            
        layers = [CausalConv( #first layer
            in_dim=input_dim,
            out_dim=channels,
            kernel_size=7 
        )]
        
        curr_channels = channels
        for s in conv_strides:
            
            #applying the 50% overlapping rule:
            k = 2*s #causes downsampling in temporal resolution.
            
            #therefore we are making up for it 
            out_channels = channels*2 
            #out_channels get continously increased by (*2)
            
            layers.append(
                CausalConv(
                    in_dim=curr_channels,
                    out_dim=out_channels,
                    stride=s,
                    kernel_size=k
                )
            )
            
            for d in dilations:
                #adding the dilation layers for every iteration in the encoder.
                layers.append(
                    CausalResBlock(
                        in_dim=curr_channels,
                        out_dim=out_channels,
                        dilation=d
                    )
                )
        
            curr_channels = out_channels #512
            #curr_channels are updated from channels to out_channels 
            
        layers.append(nn.ELU())
        layers.append(
            CausalConv(
                in_dim=curr_channels, #512
                out_dim=latent_dim, #128
                dilation=dilations 
            )
        )
        
        #defining the encoder model:
        self.model = nn.Sequential(*layers)
        
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        return self.model(x)
        
        

        
        
            
        