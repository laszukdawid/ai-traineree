import torch
from numpy import ndarray


def to_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, list) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    else:
        return torch.tensor(x)
