import pytest
import torch

from ai_traineree.agents import utils
from ai_traineree.agents.utils import compute_gae


def test_revert_norm_returns_default():
    # Assign
    rewards = torch.tensor([0, 0, 0, 1, 0, 1]).reshape(6, 1)  # Shape: (6, 1)
    dones = torch.tensor([False, False, False, True, False, False]).reshape(6, 1)  # Shape: (6, 1)
    expected = torch.tensor([-1.5628, -0.7166, 0.1382, 1.0015, 0.1382, 1.0015]).reshape(6, 1)  # Shape: (6, 1)

    # Act
    returns = utils.revert_norm_returns(rewards, dones)

    # Assert
    assert isinstance(returns, torch.Tensor)
    assert returns.shape == (6, 1)
    assert torch.allclose(returns, expected, atol=1e-4)


def test_revert_norm_returns_gamma():
    # Assign
    rewards = torch.tensor([0, -0.5, 0, 1]).reshape(4, 1)  # Shape: (4, 1)
    dones = torch.tensor([False, False, False, False]).reshape(4, 1)  # Shape: (4, 1)
    expected = torch.tensor([-0.6996, -0.9148, 0.3767, 1.2377]).reshape(4, 1)  # Shape: (4, 1)

    # Act
    returns = utils.revert_norm_returns(rewards, dones, gamma=0.5)

    # Assert
    assert isinstance(returns, torch.Tensor)
    assert returns.shape == (4, 1)
    assert torch.allclose(returns, expected, atol=1e-4)


def test_revert_norm_returns_two_dim():
    # Assign
    rewards = torch.tensor([[0, 0], [0, -0.5], [0, 0], [0, 1]])  # Shape: (4, 2)
    dones = torch.tensor([[False, False], [False, False], [False, False], [False, False]])  # Shape: (4, 2)
    expected = torch.tensor([[0, -0.6996], [0, -0.9148], [0, 0.3767], [0, 1.2377]])  # Shape: (4, 2)

    # Act
    returns = utils.revert_norm_returns(rewards, dones, gamma=0.5)

    # Assert
    assert isinstance(returns, torch.Tensor)
    assert returns.shape == (4, 2)
    assert torch.allclose(returns, expected, atol=1e-4)


def test_compute_gae_no_terminal():
    # Assign
    rewards = torch.tensor([0, -1, -1, 1]).reshape(4, 1)  # Shape: (4, 1)
    dones = torch.tensor([0, 0, 0, 0]).reshape(4, 1)  # Shape: (4, 1)
    gamma = 0.99
    tau = 0.9
    values = torch.tensor([[0], [-1.3], [-1.9], [-2.5]])  # Sape: (4, 1)
    next_value = torch.tensor([-4.])  # Shape: (1,)

    # Act
    advantages = compute_gae(rewards, dones, values, next_value, gamma, tau)

    # Assert
    assert advantages.shape == (4, 1)
    assert advantages.shape == rewards.shape
    assert isinstance(advantages, torch.Tensor)
    assert advantages.flatten().tolist() == pytest.approx([-4.27, -3.35, -1.98, -0.46], 0.1)


def test_compute_gae_terminals():
    # Assign
    rewards = torch.tensor([0, -1, 10, 0, -1]).reshape(5, 1)  # Shape: (5, 1)
    dones = torch.tensor([0, 0, 1, 0, 0]).reshape(5, 1)  # Shape: (5, 1)
    gamma = 0.99
    tau = 0.9
    values = torch.tensor([0, -1.3, 3, -1.9, -2.5]).reshape(5, 1)  # Shape: (5, 1)
    next_value = torch.tensor([-4.])  # Shape: (1,)

    # Act
    advantages = compute_gae(rewards, dones, values, next_value, gamma, tau)

    # Assert
    assert advantages.shape == (5, 1)
    assert advantages.shape == rewards.shape
    assert advantages.flatten().tolist() == pytest.approx([7.18, 9.50, 7.00, -2.77, -2.46], 0.1)


def test_compute_gae_two_dimensional_input():
    """First dim is the same as in `test_compute_gae_no_terminals`. Second dim is all zeros.
    """
    # Assign
    rewards = torch.tensor([[0, -1, -1, 1], [0, 0, 0, 0]]).T  # Shape: (4, 2)
    dones = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]]).T  # Shape: (4, 2)
    gamma = 0.99
    tau = 0.9
    values = torch.tensor([[0, -1.3, -1.9, -2.5], [0, 0, 0, 0]]).T  # Sape: (4, 2)
    next_value = torch.tensor([-4., 0])  # Shape: (2,)
    expected = torch.tensor([[-4.27, -3.35, -1.98, -0.46], [0, 0, 0, 0]]).T  # Shape: (4, 2)

    # Act
    advantages = compute_gae(rewards, dones, values, next_value, gamma, tau)

    # Assert
    assert advantages.shape == (4, 2)
    assert advantages.shape == rewards.shape
    assert torch.allclose(advantages, expected, atol=1e-2)


if __name__ == "__main__":
    test_compute_gae_two_dimensional_input()
