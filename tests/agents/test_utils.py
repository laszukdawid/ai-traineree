from ai_traineree.agents.utils import compute_gae
import pytest
import torch

from torch import tensor
from ai_traineree.agents import utils


def test_revert_norm_returns_default():
    # Assign
    rewards = [0, 0, 0, 1, 0, 1]
    dones = [False, False, False, True, False, False]
    expected = torch.tensor([-1.5628, -0.7166, 0.1382, 1.0015, 0.1382, 1.0015])

    # Act
    returns = utils.revert_norm_returns(rewards, dones)

    # Assert
    assert all(torch.isclose(returns, expected, atol=1e-4))


def test_revert_norm_returns_gamma():
    # Assign
    rewards = [0, -0.5, 0, 1]
    dones = [False, False, False, False]
    expected = tensor([-0.6996, -0.9148, 0.3767, 1.2377])

    # Act
    returns = utils.revert_norm_returns(rewards, dones, gamma=0.5)

    # Assert
    assert all(torch.isclose(returns, expected, atol=1e-4))


def test_compute_gae_no_terminal():
    # Assign
    rewards = tensor([0, -1, -1, 1])
    dones = tensor([0, 0, 0, 0])
    gamma = 0.99
    tau = 0.9
    values = tensor([0, -1.3, -1.9, -2.5])
    next_value = tensor([-4.])

    # Act
    advantages = compute_gae(rewards, dones, values, next_value, gamma, tau)

    # Assert
    assert len(advantages) == len(rewards)
    assert isinstance(advantages, torch.Tensor)
    assert advantages.tolist() == pytest.approx([-4.27, -3.35, -1.98, -0.46], 0.01)


def test_compute_gae_terminals():
    # Assign
    rewards = tensor([0, -1, 10, 0 -1, -1])
    dones = tensor([0, 0, 1, 0, 0])
    gamma = 0.99
    tau = 0.9
    values = tensor([0, -1.3, 3, -1.9, -2.5])
    next_value = tensor([-4.])

    # Act
    advantages = compute_gae(rewards, dones, values, next_value, gamma, tau)

    # Assert
    assert len(advantages) == len(rewards)
    assert advantages.tolist() == pytest.approx([7.18, 9.51, 7.00, -3.77, -2.46], 0.01)


if __name__ == "__main__":
    test_compute_gae_terminals()
