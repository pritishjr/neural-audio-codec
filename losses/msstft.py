

#implementation of the msstft loss which is important to get the optimal frequency resolutions and temporal resolution.
#calculating the loss: 

import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F

from typing import List

class STFTLOSS(nn.Module):
    #from EnCodec (same implementation)
    def __init__(
        self,
        window_fn: torch.hamming_window, #default from encodec
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 256,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window_fn = window_fn
        
        #calculating the linear spectrogram: (B,F,T_frames)
        self.spectogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=window_fn,
            normalized=False, 
            center=True,
            pad_mode="reflect",
            power=1.0
        )
        
    def forward(self, x:torch.Tensor, x_hat: torch.Tensor) -> List[torch.Tensor]:
        
        #input: (B,1,T) for both x and x_hat
        #first squeeze to (B*1, T) 
        x = x.squeeze(1)
        x_hat = x_hat.squeeze(1)
        
        #computing spectrograms: (B,F,T)
        mag_x = self.spectogram(x).clamp(1e-7)
        mag_xhat = self.spectogram(x).clamp(1e-7)
        
        #Spectral Convergence Loss:
        #frobenius norm (L2 norm for matrices)
        spectral_loss = (torch.linalg.matrix_norm(mag_x - mag_xhat))/(torch.linalg.matrix_norm(mag_x))
        
        #Log-Magnitude Loss:
        log_mag_loss = F.l1_loss((torch.log(mag_x)) - torch.log(mag_xhat))
        
        return [spectral_loss, log_mag_loss]
    
    
class MultiScaleSTFT(nn.Module):
    # computing the stft loss over multiple temporal and spectral resolutions:
    def __init__(
        self,
        n_ffts: List[int] = [1024, 2048, 512],
        hop_lengths: List[int] = [256, 512, 128],
        win_length: List[int] = [1024, 2048, 512]
    ):
        super().__init__()
        
        self.n_ffts = n_ffts
        self.hop_lengths = hop_lengths
        self.win_length = win_length
        
        assert len(n_ffts) == len(win_length) == len, \
            "STFT params array lengths should be the same."
            
        self.stftlosses = nn.Module([
            STFTLOSS(
                n_fft=a,
                hop_length=b,
                win_length=c,
            ) for a, b, c in zip(n_ffts, hop_lengths, win_length)
        ])
        
    def forward(self, x:torch.Tensor, x_hat: torch.Tensor) -> float:
        #aggregate over the different resolutions:
        total_loss = 0.0
        
        for stft_loss in len(self.stftlosses):
            total_loss += stft_loss(x,x_hat)
        
        #normalized loss:
        loss = total_loss/len(self.stftlosses) #float value
    
        return loss
    