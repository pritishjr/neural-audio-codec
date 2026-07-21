

#RVQ blocks will be stacked

import torch
import torch.nn as nn

from quantizer.quantize import VectorQuantizer

class ResidualVectorQuantizer(nn.Module):
    
    def __init__(self, num_stages: int, num_embeddings: int, embeddings_dim: int, commitment_cost: float = 0.30): #predefined params
        super().__init__()
        
        #for residual stacking layers
        self.num_stages = num_stages
        
        #stacking the VQs but so that the residual is passed to (i+1)
        #these params wont be updated during the 
        self.quantizers = nn.ModuleList([ 
            VectorQuantizer(num_embeddings, embeddings_dim, commitment_cost_weight=commitment_cost)
            for _ in range(num_stages) #loop=stacks
        ])
        
    def forward(self, z: torch.Tensor, n_active_stages: int = None):
        
        # RVQ helps in conditioning variable bitrate. no need to train separate models for different bitrates.
        # n_active_stages: for training in VBR (variable bitrate)
        
        if n_active_stages is None:
            n_active_stages = self.num_stages
        else:
            n_active_stages = min(self.num_stages, n_active_stages)
            
        #according to soundstream:
        residual = z #initially z is also null
        
        #defining final residual vectors 
        z_q_accumulated = torch.zeros_like(z) #copying the dims
        all_indices = [] #(B,T)
        total_aux_loss = torch.tensor()
        
        #loop:
        for i in range(n_active_stages):
            
            #extracting information from the quantizers.
            z_q_stage, ith_indices, ith_aux_loss = self.quantizers[i]
            
            z_q_accumulated = z_q_accumulated+z_q_stage
            residual = residual - z_q_stage
            
            all_indices.append(ith_indices)
            total_aux_loss = total_aux_loss+ith_aux_loss
            
        #indices for all stacks: (B,T,n_active_stages)
        stacked_indices = torch.stack(all_indices,dim=-1)
        
        return z_q_accumulated,stacked_indices,total_aux_loss
    

            