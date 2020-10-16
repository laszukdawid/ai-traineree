import numpy
import torch


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_list(e):
    if isinstance(e, torch.Tensor) or isinstance(e, numpy.ndarray):
        return e.tolist()
    elif e is None:
        return None
    else:
        return list(e)
