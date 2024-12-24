import torch

from ai_traineree.policies import MultivariateGaussianPolicy, MultivariateGaussianPolicySimple


def test_multi_gauss_simple_defaults():
    # Assign
    size = 5
    std_init = 0.5
    test_loc = torch.zeros((1, size))  # Shape: (1, 5)
    policy = MultivariateGaussianPolicySimple(size, std_init=std_init)

    # Act
    samples = policy(test_loc)

    # Assert
    assert policy.std_min == 0.0001
    assert policy.std_max == 2.0
    assert policy.param_dim == 1
    assert all(policy.std == torch.ones(size) * std_init)  # 1D
    assert samples.shape == (1, size)


def test_multi_gauss_simple_std_updates():
    # Assign
    size = 5
    std_init, std_min, std_max = 2, 0.01, 10
    test_loc = torch.arange(size).unsqueeze(0)  # Shape: (1, 5)
    test_std = torch.full((size,), std_init)  # Shape: (5,)
    # test_cov_mat = torch.eye(size).unsqueeze(0) * std_init**2  # Shape: (1, 5, 5)
    policy = MultivariateGaussianPolicySimple(size, std_init=std_init, std_min=std_min, std_max=std_max)

    # Act
    samples = policy(test_loc)

    # Assert
    assert policy.std_min == std_min
    assert policy.std_max == std_max
    assert all(policy.std == test_std)  # 1D
    assert samples.shape == (1, size)


def test_multi_gauss_simple_statistic():
    """Compares statistical measures. Might (very rarely) fail."""
    # Assign
    size = 3
    batch_size = 3000
    expected_loc = torch.tensor([-2.0, 0, 2.0])  # Shape: (3,)
    expected_std = test_std = torch.tensor([0.1, 1, 2.0])
    policy = MultivariateGaussianPolicySimple(size, std_max=max(expected_std) * 2)
    policy.std.data = test_std
    test_loc = expected_loc.repeat((batch_size, 1))  # Shape: (3000, 3)

    # Act
    samples = policy(test_loc)

    # Assert
    assert samples.shape == (batch_size, size)
    assert torch.all(torch.isclose(samples.std(dim=0), expected_std, atol=0.1))  # +/- 0.1
    assert torch.all(torch.abs(samples.mean(dim=0) - expected_loc) < 0.2 * expected_std)


def test_multi_gauss():
    # Assign
    size = 3
    policy = MultivariateGaussianPolicy(size)
    loc = torch.zeros((1, size))  # Shape: (1, 3)
    std = torch.ones((1, size))  # Shape: (1, 3)
    x = torch.hstack((loc, std))  # Shape: (1, 6)
    test_cov_mat = torch.eye(size).unsqueeze(0)  # Shape: (1, 3, 3)

    # Act
    samples = policy(x)

    # Assert
    assert policy.param_dim == 2
    assert x.shape == (1, policy.param_dim * size)  # Shape: (1, 6)
    assert samples.shape == (size,)

    assert torch.all(policy._last_dist.loc == loc)
    assert torch.all(policy._last_dist.covariance_matrix == test_cov_mat)


if __name__ == "__main__":
    test_multi_gauss_simple_statistic()
