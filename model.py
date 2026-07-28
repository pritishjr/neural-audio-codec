
#combining the architectures:

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

#importing modules:
from convolution.convblocks import CausalConv, CausalResBlock, CausalConvTranspose
from encoder.encoder import Encoder
from decoder.decoder import Decoder
from quantizer.rvq import ResidualVectorQuantizer

class Codec(nn.Model):
    
    def __init__(
        self,
        input_channels: int = 1, #mono=1, stereo=2
        channels: int = 32, #feature map 
        latent_dim: int = 128, #embeddings_dim
        strides: list[int]=[2,4,5,8],
        dilations: list[int] = [1,3,9],
        num_stages: int = 8, #rvq
        num_embeddings: int = 1024,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        
        #encoder
        self.encoder = Encoder(
            conv_strides=strides,
            input_dim=input_channels,
            channels=channels,
            latent_dim=latent_dim,
            dilations=dilations
        )
        
        #rvq
        self.rvq = ResidualVectorQuantizer(
            num_stages,
            num_embeddings,
            embeddings_dim=latent_dim,
            commitment_cost = commitment_cost
        )
        
        #decoder
        self.decoder = Decoder(
            out_channels=input_channels,
            channels=channels,
            latent_dim=latent_dim,
            strides=strides,
            dilations=dilations,
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        #x: (B,1,T)
        z_out = self.encoder(z) #(B, D, T/hop_size)
        
        #from the rvq, we had changed the sizes: (B,T',D)
        z_out = z_out.transpose(1,2)
        
        z_quantised, codebook_indices, total_loss = self.rvq(z_out) #quantized
        
        #restructuring: (B,D,T')
        z_quantised = z_quantised.transpose(1,2)
        
        z_decoded = self.decoder(z_quantised)
        
        #returns the reconstructed signal from the audio chunk.
        return z_decoded, codebook_indices, total_loss
        
