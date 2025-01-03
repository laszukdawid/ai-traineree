import math

import torch

from ai_traineree import DEVICE
from ai_traineree.types.primitive import FeatureType


class GaussianNoise:
    def __init__(self, shape: FeatureType, mu=0.0, sigma=1.0, scale=1.0, device=None):
        self.shape = shape
        self.mu = torch.zeros(size=shape) + mu  # type: ignore
        self.std = torch.zeros(size=shape) + math.sqrt(sigma)  # type: ignore
        self.scale = scale
        self.device = device if device is not None else DEVICE

    def sample(self):
        return self.scale * torch.normal(self.mu, self.std).to(self.device)


class OUProcess:
    """
    Generating Ornstein–Uhlenbeck process as an additive noise.

    Note: Although it supports tensor output it is a univariate process.

    Deriv of Weiner process is assumed to be white (uniform) noise.
    https://en.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
    """

    def __init__(self, shape: int, scale=0.2, mu=0.0, theta=0.15, sigma=0.2, device=None):
        """
        `theta` denotes the feedback factor. The smaller it is, the bigger drift is observed.
        """
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.x = torch.ones(shape, device=device, requires_grad=False) * self.mu
        self.device = device

    def reset_states(self):
        self.x[:] = self.mu

    def sample(self):
        x = self.x
        dx = self.theta * (self.mu - x) + self.sigma * torch.rand(self.x.shape, out=self.x)
        self.x = x + dx
        return self.scale * self.x
