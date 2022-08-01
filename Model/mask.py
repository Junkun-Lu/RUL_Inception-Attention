import torch
import math
import numpy as np

#  -------------- This mask use for Full-Attention ----------------------------

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


#  -------------- This mask use for ProbSparse-Attention ----------------------------
class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                            torch.arange(H)[None, :, None],
                            index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

#  -------------- This mask use for Local-Attention ----------------------------
class LocalMask():
    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            window_size = math.ceil(1.2*np.log2(L))   # mask大小      
            mask = torch.ones((L, L)).to(device)
            mask = torch.triu(mask,-window_size).T
            mask = torch.triu(mask,0).T        
            mask = mask ==0
            mask = torch.unsqueeze(mask, 0)
            self._mask = mask.expand(B, 1, L, L).to(device) 
    @property    
    def mask(self):
        return self._mask