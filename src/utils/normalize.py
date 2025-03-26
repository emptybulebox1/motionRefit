import numpy as np
import torch

def set_up_normalization(device='cpu', seq_len=16, scale=3, norm_path='../data/norm_scaled.npy'):
    assert scale==3, 'Currently only support scale=3'

    # return normalizeation function with given device
    norm = np.load(norm_path, allow_pickle=True).item()[(seq_len, scale)]
    min_val = torch.tensor(norm[0]).to(device, dtype=torch.float32)
    max_val = torch.tensor(norm[1]).to(device, dtype=torch.float32)
    
    # input: joints (..., 28*3) or (..., 28, 3)
    def denormalize(data: torch.Tensor):
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        data = (data + 1.) * (max_val - min_val) / 2. + min_val
        data = data.reshape(shape_orig)
        return data
    
    def normalize(data: torch.Tensor):
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        data = -1. + 2. * (data - min_val) / (max_val - min_val)
        data = data.reshape(shape_orig)
        return data
    
    return normalize, denormalize