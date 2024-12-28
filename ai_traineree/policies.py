from functools import lru_cache

import torch
import torch.nn as nn
from torch.distributions import Beta, Dirichlet, MultivariateNormal, Normal
from torch.distributions.distribution import Distribution

from ai_traineree.networks import NetworkType
from ai_traineree.networks.bodies import FcNet
from ai_traineree.types import FeatureType
from ai_traineree.types.dataspace import DataSpace


class PolicyType(NetworkType):
    param_dim: int


class MultivariateGaussianPolicySimple(PolicyType):
    """
    Multivariate Gaussian (Normal) Policy.

    Simplicity of this class, compared to `MultivariateGaussianPolicy`, is in
    the assumption that the covariance is diagonal, sample independent and
    is treated a trainable parameter.
    """

    param_dim = 1

    def __init__(
        self, size: int, std_init: float = 0.5, std_min: float = 0.0001, std_max: float = 2.0, device=None, **kwargs
    ):
        """
        Parameters:
            size (int): Size of the observation.
            std_init: Initial value for covariance's diagonal. All values start the same. Default: 0.5.
            std_min: Minimum value for standard deviation. Default: 0.0001.
            std_max: Maximum value for standard deviation. Default: 2.
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
    @lru_cache(maxsize=10)
    def _empty_std(batch_size: int, size: int, device):
        return torch.zeros((batch_size, size, size), device=device)

    @staticmethod
    @lru_cache(maxsize=10)
    def diag_idx(batch_size: int, size: int, device):
        return torch.arange(size, device=device).repeat((batch_size, 1, 1))

    def forward(self, x, deterministic: bool = False) -> Distribution:
        """Samples from distribution.

        Parameters:
            x (tensor): Uses a location (mu) for the distrubition.
            deterministic (bool): Whether to sample from distribution, or use estimates.
                Default: False, i.e. it'll sample distribution.

        """
        if deterministic:
            return x.view(-1, 1)

        x = x.float()  # Need to convert as dist doesn't work on int
        self.std.data.clamp_(self.std_min, self.std_max)

        if self.size == 1:
            return self.dist(x.view(-1, 1), scale=self.std.view(-1, 1))

        batch_size = x.shape[0]
        new_shape = (batch_size, 1, 1)
        idx = self.diag_idx(batch_size, self.size, device=x.device)
        std_shell = self._empty_std(batch_size, self.size, x.device)
        std_vals = self.std.repeat(new_shape)
        std = std_shell.scatter(1, idx, std_vals)
        self._last_samples = {"mu": x, "std": self.std}
        self._last_dist = self.dist(x, scale_tril=std)
        return self._last_dist.rsample()

    def act(self, x):
        return self.forward(x, deterministic=True)

    def log_prob(self, samples) -> torch.Tensor:
        assert self._last_dist is not None, "Need to execute `forward` on data first"
        return self._last_dist.log_prob(samples)


class MultivariateGaussianPolicy(PolicyType):
    """
    Multivariate Gaussian (Normal) Policy.

    In contrast to `MultivariateGaussianPolicySimple` it assumes that
    distribution's characteristics are estimated by the network rather
    than optimized by the optimizer.
    Both location and covariance are assumed to be inputs into the policy.

    """

    param_dim = 2

    def __init__(
        self, size: int, std_init: float = 1.0, std_min: float = 0.001, std_max: float = 2.0, device=None, **kwargs
    ):
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
        self._last_samples = None

    @staticmethod
    @lru_cache(maxsize=10)
    def _empty_std(batch_size: int, size: int, device):
        return torch.zeros((batch_size, size, size), device=device)

    @staticmethod
    @lru_cache(maxsize=10)
    def diag_idx(batch_size: int, size: int, device):
        return torch.arange(size, device=device).repeat((batch_size, 1, 1)).view(batch_size, size, 1)

    def forward(self, x, deterministic=False) -> Distribution:
        """Returns distribution"""
        x = x.view(-1, self.param_dim, self.size)
        mu = x[:, 0]
        if deterministic:
            return mu.view(-1, 1)

        std = torch.clamp(x[:, 1], self.std_min, self.std_max).unsqueeze(-1)

        if self.size == 1:
            _mu = mu.view(-1, 1)
            _scale = std.view(-1, 1)
            self._last_samples = {"mu": _mu, "std": torch.diagonal(_scale)}
            self._last_dist = self.dist(mu.view(-1, 1), scale=_scale)
            return self._last_dist.rsample()

        batch_size = x.shape[0]
        std_shell = self._empty_std(batch_size, self.size, x.device)
        idx = self.diag_idx(batch_size, self.size, x.device)
        std = std_shell.scatter(-1, idx, std)
        _std_values = torch.diagonal(std.squeeze(0))
        self._last_samples = {"mu": mu.squeeze(0), "std": _std_values}
        self._last_dist = self.dist(mu.squeeze(0), std.squeeze(0))

        return self._last_dist.rsample()

    def act(self, x) -> torch.Tensor:
        """Deterministic pass. Ignores covariance and returns locations directly."""
        return x.view(-1, self.size, self.param_dim)[..., 0]

    def log_prob(self, samples):
        return self._last_dist.log_prob(samples)


class GaussianPolicy(PolicyType):
    """
    Univariate Gaussian (Normal) Distribution.
    Has two heads; one for location estimate and one for standard deviation.
    """

    def __init__(self, in_features: FeatureType, out_features: FeatureType, out_scale: float = 1, **kwargs):
        """
        Parameters:
            size: Observation's dimensionality upon sampling.

        """
        super(GaussianPolicy, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.out_scale = out_scale

        hidden_layers = kwargs.get("hidden_layers")
        self.dist = Normal
        self.mu = FcNet(in_features, out_features, hidden_layers=hidden_layers, **kwargs)
        self.log_std = FcNet(in_features, out_features, hidden_layers=hidden_layers, **kwargs)

        self.log_std_min = -10
        self.log_std_max = 2

        self._last_dist: Distribution | None = None
        self._last_samples = None

    def log_prob(self, samples) -> torch.Tensor | None:
        if self._last_dist is None:
            return None
        return self._last_dist.log_prob(samples).sum(axis=-1)

    def forward(self, x, deterministic: bool = False) -> torch.Tensor:
        mu = self.mu(x)
        if deterministic:
            self._last_dist, self._last_samples = (None, None)
            return mu

        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        self._last_dist = dist = self.dist(mu, std)
        actions = dist.rsample()
        self._last_samples = {"mu": mu.squeeze(0), "std": std}
        return actions


class BetaPolicy(PolicyType):
    """
    Multivarate generalized version of the Dirichlet (1D) distribution.

    Uses torch.distributions.Beta or torch.distributions.Dirichlet
    distirubitions depending on the input size.

    https://pytorch.org/docs/stable/distributions.html#beta
    https://pytorch.org/docs/stable/distributions.html#dirichlet
    """

    param_dim = 2

    # def __init__(self, size: int, bounds: Tuple[float, float] = (1, float("inf"))):
    def __init__(self, size: int, bound_space: DataSpace, out_scale: float = 1, **kwargs):
        """
        Parameters:
            size: Observation's dimensionality upon sampling.
            bounds: Beta dist input clamp for both alpha and betas.
                Both concentration are expected to be larger than 1.

        """
        super(BetaPolicy, self).__init__()
        if bound_space.low is None or bound_space.high is None:
            raise ValueError(
                "Bound space needs to have both low and high boundaries. "
                f"Provided: low={bound_space.low}, high={bound_space.high}"
            )
        self.bound_space = bound_space
        self.action_size = size
        self.dist = Beta if size > 1 else Dirichlet
        self._last_dist: Distribution | None = None
        self._last_samples = None

    def forward(self, x, deterministic: bool = False) -> torch.Tensor:
        x = x.view(-1, self.action_size, self.param_dim)
        loc = x[..., 0]
        if deterministic:
            return loc

        # x = torch.clamp(x, self.bound_space.low, self.bound_space.high)
        # self._last_dist = self.dist(x[..., 0], x[..., 1])
        self._last_dist = self.dist()
        return self._last_dist.rsample()

    def log_prob(self, samples):
        if self._last_dist is None:
            return None
        return self._last_dist.log_prob(samples).mean(dim=-1)


class DirichletPolicy(PolicyType):
    param_dim = 2

    def __init__(self, size: int, *, alpha_min: float = 0.05):
        super(DirichletPolicy, self).__init__()
        self.size = size
        self.alpha_min = alpha_min

    def forward(self, x, deterministic: bool = False) -> torch.Tensor:
        _x = x.view(-1, self.param_dim, self.size)
        loc = _x[..., 0]
        if deterministic:
            return loc
        alpha = torch.clamp(_x[..., 1], self.alpha_min)
        self._last_dist = Dirichlet(alpha)
        self._last_loc = loc.detach()
        return loc + self._last_dist.rsample()

    def log_prob(self, samples) -> torch.Tensor:
        _sampled = samples - self._last_loc
        return self._last_dist.log_prob(_sampled)


class DeterministicPolicy(PolicyType):
    param_dim = 1

    def __init__(self, action_size):
        super(DeterministicPolicy, self).__init__()
        self.action_size = action_size

    def forward(self, x):
        return x
