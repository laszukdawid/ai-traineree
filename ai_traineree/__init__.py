import torch

__version__ = "0.4.2"


# This is expected to be safe, although in PyTorch 1.7 it comes as a warning,
# if CUDA is not installed.
try:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except Exception:
    DEVICE = torch.device("cpu")
