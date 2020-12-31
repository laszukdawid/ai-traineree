import torch
import torch.nn as nn
import random

from ai_traineree.networks import NetworkType
from functools import lru_cache
from torch.distributions import Beta, Dirichlet, MultivariateNormal, Normal
from torch.distributions.distribution import Distribution

from typing import Tuple


class PolicyType(NetworkType):
    param_dim: int


class MultivariateGaussianPolicySimple(PolicyType):
    """
    Multivariate Gaussian (Normal) Policy.

    Simplicity of this class, compared to `MultivariateGaussianPolicy`, is due to
    the assumption that the covariance is sample independent and it is a trainable
    parameter.
    """

    param_dim = 1

    def __init__(self, size: int, std_init: float=1.0, std_min: float=0.1, std_max: float=3., device=None):
        """
        Parameters:
            size: Size of the observation.
            batch_size: Expected size of batch. Helps in memory assignments.
            std_init: (default 2) Initial value for covariance's diagonal. All values start the same.
            std_min: Minimum value for standard deviation.
            std_max: Maximum value for standard deviation.
            device: Device where to allocate memory. CPU or CUDA.

        """
        super(MultivariateGaussianPolicySimple, self).__init__()
        self.size = size
        self.dist = Normal if size == 1 else MultivariateNormal
        self.std_min = std_min
        self.std_max = std_max
        std_init = float(max(min(std_max, std_init), std_min))
        self.std = nn.Parameter(torch.full((self.size,), std_init, device=device))

    @staticmethod
    @lru_cache
    def _empty_std(batch_size: int, size: int, device):
        return torch.zeros((batch_size, size, size), device=device)

    @staticmethod
    @lru_cache
    def diag_idx(batch_size: int, size: int, device):
        return torch.arange(size, device=device).repeat((batch_size, 1, 1))

    def forward(self, x) -> Distribution:
        """Returns distribution"""
        self.std.data = torch.clamp(self.std, self.std_min, self.std_max)
        if self.size == 1:
            return self.dist(x.view(-1, 1), scale=self.std.view(-1, 1))

        # Distinction here is primarily performance optimization (though it isn't too optimal)
        batch_size = x.shape[0]
        if len(x.shape) == 1 or x.shape[0] == 1:
            new_shape = (1, self.size, 1)
            idx = torch.arange(self.size, device=x.device).view(new_shape)
            std = self._empty_std(batch_size, self.size, x.device).scatter(-1, idx, self.std.repeat(new_shape))
        else:
            std = self.std.repeat((batch_size, 1, 1))
            std = self._empty_std(batch_size, self.size, x.device).scatter(1, self.diag_idx(batch_size, self.size, x.device), std)
        return self.dist(x, scale_tril=std)

    def act(self, x):
        return x

    @staticmethod
    def log_prob(dist, samples) -> torch.Tensor:
        return dist.log_prob(samples)


class MultivariateGaussianPolicy(PolicyType):
    """
    Multivariate Gaussian (Normal) Policy.

    In contrast to `MultivariateGaussianPolicySimple` it assumes that
    distribution's characteristics are estimated by the network rather
    than optimized by the optimizer.
    Both location and covariance are assumed to be inputs into the policy.

    """

    param_dim = 2

    def __init__(self, size: int, std_init: float=1.0, std_min: float=0.1, std_max: float=3., device=None):
        """
        Parameters:
            size: Observation's dimensionality upon sampling.
            batch_size: Expected size of batch.
            device: Device where to allocate memory. CPU or CUDA.
        """
        super(MultivariateGaussianPolicy, self).__init__()
        self.size = size
        self.dist = Normal if size == 1 else MultivariateNormal
        self.std_init = std_init
        self.std_min = std_min
        self.std_max = std_max

    @staticmethod
    @lru_cache
    def _empty_std(batch_size: int, size: int, device):
        return torch.zeros((batch_size, size, size), device=device)

    @staticmethod
    @lru_cache
    def diag_idx(batch_size: int, size: int, device):
        return torch.arange(size, device=device).repeat((batch_size, 1, 1)).view(batch_size, size, 1)

    def forward(self, x) -> Distribution:
        """Returns distribution"""
        x = x.view(-1, self.param_dim, self.size)
        mu = x[:, 0]
        std = torch.clamp(x[:, 1], self.std_min, self.std_max).unsqueeze(-1)
        if self.size == 1:
            return self.dist(mu.view(-1, 1), scale=std.view(-1, 1))

        batch_size = x.shape[0]
        if x.shape[0] == 1:
            idx = torch.arange(self.size, device=x.device).view(1, self.size, 1)
            std = torch.zeros((1, self.size, self.size), device=x.device).scatter(-1, idx, std)
        else:
            std = self._empty_std(batch_size, self.size, x.device).scatter(-1, self.diag_idx(batch_size, self.size, x.device), std)
        return self.dist(mu, scale_tril=std)

    def act(self, x) -> torch.Tensor:
        """Deterministic pass. Ignores covariance and returns locations directly."""
        return x.view(-1, self.size, self.param_dim)[..., 0]

    @staticmethod
    def log_prob(dist, samples) -> torch.Tensor:
        return dist.log_prob(samples)


class GaussianPolicy(PolicyType):
    """
    Univariate Gaussian (Normal) Distribution.
    """

    param_dim = 2

    def __init__(self, size: int):
        """
        Parameters:
            size: Observation's dimensionality upon sampling.

        """
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

    def log_prob(self, dist: Normal, samples: torch.Tensor) -> torch.Tensor:
        return dist.log_prob(samples).mean(dim=-1)


class BetaPolicy(PolicyType):
    """
    Multivarate generalized version of the Dirichlet (1D) distribution.

    Uses torch.distributions.Beta or torch.distributions.Dirichlet
    distirubitions depending on the input size.

    https://pytorch.org/docs/stable/distributions.html#beta
    https://pytorch.org/docs/stable/distributions.html#dirichlet
    """

    param_dim = 2

    def __init__(self, size: int, bounds: Tuple[float, float]=(1, float('inf'))):
        """
        Parameters:
            size: Observation's dimensionality upon sampling.
            bounds: Beta dist input clamp for both alpha and betas.
                Both concentration are expected to be larger than 1.

        """
        super(BetaPolicy, self).__init__()
        self.bounds = bounds
        self.action_size = size
        self.dist = Beta if size > 1 else Dirichlet

    def forward(self, x) -> Distribution:
        x = x.view(-1, self.action_size, self.param_dim)
        x = torch.clamp(x, self.bounds[0], self.bounds[1])
        dist = self.dist(x[..., 0], x[..., 1])
        return dist

    @staticmethod
    def log_prob(dist, samples):
        return dist.log_prob(samples).mean(dim=-1)


class DirichletPolicy(PolicyType):

    param_dim = 1

    def __init__(self, *, alpha_min: float=0.05):
        super(DirichletPolicy, self).__init__()
        self.alpha_min = alpha_min

    def forward(self, x) -> Distribution:
        x = torch.clamp(x, self.alpha_min)
        return Dirichlet(x)

    def log_prob(self, dist: Dirichlet, samples) -> torch.Tensor:
        return dist.log_prob(samples)


class DeterministicPolicy(PolicyType):

    param_dim = 1

    def __init__(self, action_size):
        super(DeterministicPolicy, self).__init__()
        self.action_size = action_size

    def forward(self, x):
        return x
