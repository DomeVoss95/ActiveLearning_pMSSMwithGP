import torch 
import numpy as np

def normalize(data, data_min=None, data_max=None):
    if isinstance(data, np.ndarray):
        data = torch.tensor(data)  # Convert NumPy array to PyTorch tensor

    if data_min is None:
        data_min = data.min(dim=0, keepdim=True).values
    if data_max is None:
        data_max = data.max(dim=0, keepdim=True).values
    return (data - data_min) / (data_max - data_min), data_min, data_max