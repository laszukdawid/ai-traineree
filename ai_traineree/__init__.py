import numpy
import torch


# This is expected to be safe, although in PyTorch 1.7 it comes as a warning,
# if CUDA is not installed.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_list(e):
    if isinstance(e, torch.Tensor) or isinstance(e, numpy.ndarray):
        return e.tolist()
    elif e is None:
        return None
    else:
        try:
            return list(e)
        except Exception:
            return [e]
