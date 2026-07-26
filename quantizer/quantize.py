
#mathematical modelling of the quantizing algorithm.
#(bits and pieces taken from kyutai-mimi blog)

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    #params: embeddings_dim(D), num_embeddings(K) - codebook
    def __init__(self, num_embeddings: int, embeddings_dim: int, commitment_cost_weight: int):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.commitment_cost_weight = commitment_cost_weight
        
        #defining a codebook tensor: empty weight matrix since it is learnable.
        # C - (K,D)
        self.codebook = nn.Parameter(torch.zeros(num_embeddings,embeddings_dim))
        
    
    def forward(self, z: torch.Tensor): #forward pass
        
        """
        z: (B,T,D) - batch, time frames, latent dims
        this is the 
        """
        #implementing the math:
        
        #initializing shape of the z tensor
        B, T, D = z.shape
        
        #flatten into 2D for euclidean comp:
        z_flat = z.view(-1, D) #new shape: (B*T, D)
        
        #defining z^2
        z_sq = torch.sum(
            z_flat**2, 
            dim= 1, #becomes single dim
            keepdim=True #tensor is returned
        )
        
        #defining kth element (K, ) for the 
        c_sq = torch.sum(
            self.codebook**2,
            dim=1,
            keepdim=True
        )
        
        #middle term of the expansion:
        mid_z_c = torch.matmul( 
            z_flat,
            self.codebook.t() #transposed C
        )
        mid_term = 2*mid_z_c
        
        #final expression:
        distances = z_sq - mid_term + c_sq

        #finding the index of the codeboook with the least distance from 
        k_star = torch.argmin(distances, dim=1)
        
        #gathering the k_star.th element
        z_q_flat = self.codebook[k_star]
        
        #unflatten the vector:
        z_q = z_q_flat.view(B, T, D)
        
        '''
        the encoder: a fxn of argmin- which is a step function. therefore all of its derivative- the gradient calculation for backprop- will be 0.
        we still want the encoder to learn when training it.
        therefore we fake it with- straight-through estimator (STE) which we apply to the z-q.
        we also detach it (using .detach())- detaches from the autograd engine. no gradient updates for this vector in the training loop. that term becomes a CONSTANT basically. 
        '''
        #STE: (from kyutai - mimi)
        z_q = z + (z_q - z).detach()
        
        #updating codebook to match the vector
        codebook_loss = F.mse_loss(z_q, z.detach())
        
        #updating the vector to match the codebook element
        commitment_loss = F.mse_loss(z_q.detach(), z)
        
        #computing total loss:
        aux_loss = codebook_loss + (self.commitment_cost_weight * commitment_loss)

        return z_q, k_star.view(B,T), aux_loss
        #OUTPUT:
        #residual_vector, indices, loss
    

        
        
        