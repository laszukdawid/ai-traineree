from ai_traineree.networks import NetworkType
import torch
from torch.distributions import Beta, MultivariateNormal, Normal

from typing import Tuple


class PolicyType(NetworkType):

    param_dim: int


class GaussianPolicy(PolicyType):

    param_dim = 2

    def __init__(self, size):
        super(GaussianPolicy, self).__init__()
        self.action_size = size
        # self.dist = Normal if size == 1 else MultivariateNormal
        self.dist = Normal
        self.std_min = 0.001
        self.std_max = 5

    def forward(self, x):
        """Returns distribution"""
        x = x.view(-1, self.action_size, self.param_dim)
        mu = x[..., 0]
        std = torch.clamp(x[..., 1], self.std_min, self.std_max)
        return self.dist(mu, std)


class BetaPolicy(PolicyType):

    param_dim = 2

    def __init__(self, acton_size, bounds: Tuple[float, float]):
        super(BetaPolicy, self).__init__()
        self.bounds = bounds
        self.action_size = acton_size

    def forward(self, x):
        x = x.view(-1, self.action_size, self.param_dim)
        x = torch.clamp(x, 1)
        dist = Beta(x[..., 0], x[..., 1])
        return dist


class DeterministicPolicy(PolicyType):

    param_dim = 1

    def __init__(self, action_size):
        super(DeterministicPolicy, self).__init__()
        self.action_size = action_size

    def forward(self, x):
        return x
