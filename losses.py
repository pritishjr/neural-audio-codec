

import torch
import torch.nn as nn
import torch.nn.functional as F

#initialising the loss functions: hinge losses for discrinators and generators:
def hinge_loss_discriminator(disc_fake: torch.Tensor, disc_real: torch.Tensor):
    
    #by definition: for 1D
    loss_fake = torch.mean(F.relu(1.0 - disc_fake))
    loss_real = torch.mean(F.relu(1.0 - disc_real))
    
    return loss_fake + loss_real

def hinge_loss_generator(disc_fake:torch.Tensor):
    
    #by definition:
    loss_real = - torch.mean(F.relu(1.0 - disc_fake))
    
def feature_mapping_loss(fmap_real: list[torch.Tensor], fmap_fake: list[torch.Tensor]):
    #L1 - manhattan distance (magnitude) between the discriminator and generated pair.
    
    loss = 0.0
    for fmap_r, fmap_f in zip(fmap_real, fmap_fake):
        loss += F.l1_loss(fmap_r.detach(), fmap_f) 
        #important to detach the fmap_r from the computation graph of pytorch to save memory since we would not be needing its gradient. loss is calculated only in the fake(generated) feature map.
        
    #normalising
    loss = loss/len(fmap_real)
    
    return loss