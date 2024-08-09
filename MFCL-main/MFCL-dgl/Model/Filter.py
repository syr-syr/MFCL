"""
Time: 2024.4.11
Author: Yiran Shi
Function: FFT and IFFT
"""
import torch

def Filter(emb):
    N, L = emb.size()
    tmp = torch.fft.rfft(emb, dim=1, norm='ortho')
    N_fft, L_fft = tmp.size()
    if L_fft < L:
        tmp = torch.nn.functional.pad(tmp, (0, L - L_fft))
    elif L_fft > L:
        tmp = tmp[:, :, :L]
    return tmp

def IFilter(emb):
    w_emb = torch.fft.irfft(emb, dim=1, n=1, norm='ortho')
    return w_emb

