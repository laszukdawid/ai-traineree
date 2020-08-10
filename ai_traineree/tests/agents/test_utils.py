import numpy as np
import torch

from ai_traineree.agents import utils


def test_to_np():
    # Assign
    test = torch.tensor([0, 1, 2, 3], device="cpu")
    expected = np.array([0, 1, 2, 3])

    # Act
    out = utils.to_np(test)

    # Assert
    assert all(out == expected)


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
    expected = torch.tensor([-0.6996, -0.9148, 0.3767, 1.2377])

    # Act
    returns = utils.revert_norm_returns(rewards, dones, gamma=0.5)

    # Assert
    assert all(torch.isclose(returns, expected, atol=1e-4))
