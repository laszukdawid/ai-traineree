import torch
import numpy as np
from typing import Union, Sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GaussianNoise:
    def __init__(self, shape: Union[int, Sequence[int]], mu=0., sigma=1., scale=1., device=None):
        self.shape = shape
        self.mu = mu
        self.sigma = sigma
        self.scale = scale
        self.device = device if device is not None else DEVICE

    def sample(self):
        return torch.tensor(self.scale * np.random.normal(self.mu, self.sigma, self.shape)).to(self.device)
