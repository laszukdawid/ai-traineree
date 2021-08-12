import numpy
import torch

__version__ = "0.2.0"


# This is expected to be safe, although in PyTorch 1.7 it comes as a warning,
# if CUDA is not installed.
try:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except Exception:
    DEVICE = torch.device("cpu")


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
