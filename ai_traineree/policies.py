from torch import Tensor
from torch.distributions.distribution import Distribution
from ai_traineree.networks import NetworkType
import torch
from torch.distributions import Beta, Dirichlet, MultivariateNormal, Normal

from typing import Tuple


class PolicyType(NetworkType):
    param_dim: int


class MultivariateGaussianPolicy(PolicyType):

    param_dim = 2

    def __init__(self, size: int, batch_size: int, device=None):
        super(MultivariateGaussianPolicy, self).__init__()
        self.size = size
        self.dist = Normal if size == 1 else MultivariateNormal
        # self.dist = Normal
        self.std_min = 0.001
        self.std_max = 5
        self.batch_size = batch_size
        idx = torch.arange(size, device=device).unsqueeze(1)
        self._empty_std = torch.zeros((batch_size, size, size), device=device)
        self.diag_idx = torch.stack((idx,)*batch_size)

    def forward(self, x) -> Distribution:
        """Returns distribution"""
        x = x.view(-1, self.size, self.param_dim)
        mu = x[..., 0]
        std = torch.clamp(x[..., 1], self.std_min, self.std_max).unsqueeze(-1)
        if x.shape[0] == 1:
            idx = torch.arange(self.size, device=x.device).view(1, self.size, 1)
            std = torch.zeros((1, self.size, self.size), device=x.device).scatter(-1, idx, std)
        else:
            std = self._empty_std.scatter(-1, self.diag_idx, std)
        return self.dist(mu, scale_tril=std)

    def log_prob(self, dist, samples) -> Tensor:
        return dist.log_prob(samples)


class GaussianPolicy(PolicyType):

    param_dim = 2

    def __init__(self, size: int):
        super(GaussianPolicy, self).__init__()
        self.action_size = size
        self.dist = Normal
        self.std_min = 0.001
        self.std_max = 5

    def forward(self, x) -> Distribution:
        x = x.view(-1, self.action_size, self.param_dim)
        mu = x[..., 0]
        std = torch.clamp(x[..., 1], self.std_min, self.std_max)
        return self.dist(mu, std)

    def log_prob(self, dist: Normal, samples: Tensor) -> Tensor:
        return dist.log_prob(samples).mean(dim=-1)


class BetaPolicy(PolicyType):

    param_dim = 2

    def __init__(self, size, bounds: Tuple[float, float]):
        super(BetaPolicy, self).__init__()
        self.bounds = bounds
        self.action_size = size
        self.dist = Beta if size == 1 else Dirichlet

    def forward(self, x) -> Distribution:
        x = x.view(-1, self.action_size, self.param_dim)
        x = torch.clamp(x, 1)
        dist = self.dist(x[..., 0], x[..., 1])
        return dist

    def log_prob(self, dist: Beta, samples):
        return dist.log_prob(samples).mean(dim=-1)


class DirichletPolicy(PolicyType):

    param_dim = 1

    def __init__(self, *, alpha_min: float=0.05):
        super(DirichletPolicy, self).__init__()
        self.alpha_min = alpha_min

    def forward(self, x) -> Distribution:
        x = torch.clamp(x, self.alpha_min)
        return Dirichlet(x)

    def log_prob(self, dist: Dirichlet, samples) -> Tensor:
        return dist.log_prob(samples)


class DeterministicPolicy(PolicyType):

    param_dim = 1

    def __init__(self, action_size):
        super(DeterministicPolicy, self).__init__()
        self.action_size = action_size

    def forward(self, x):
        return x
