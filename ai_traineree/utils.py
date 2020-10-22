import torch
from numpy import ndarray


def to_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, ndarray):
        return torch.from_numpy(x)
    else:
        return torch.tensor(x)
