import torch

__version__ = "0.5.0"


try:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except Exception:
    DEVICE = torch.device("cpu")
