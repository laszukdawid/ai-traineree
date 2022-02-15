import torch

__version__ = "0.4.4"


try:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except Exception:
    DEVICE = torch.device("cpu")
