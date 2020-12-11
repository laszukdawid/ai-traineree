import torch
import torch.nn as nn

from torch import Tensor


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)  # type: ignore


def hard_update(target: nn.Module, source: nn.Module):
    """Updates one network based on another."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)  # type: ignore


def compute_gae(rewards: Tensor, dones: Tensor, values: Tensor, next_value: Tensor, gamma=0.99, lamb=0.9) -> Tensor:
    """Uses General Advantage Estimator to compute... general advantage estiomation."""
    _tmp_values = torch.cat((values, next_value[None, ...]))
    masks = 1 - dones.int()
    gaes = torch.zeros_like(_tmp_values)
    deltas = rewards + gamma * _tmp_values[1:] * masks - _tmp_values[:-1]
    for idx in reversed(range(len(rewards))):
        gaes[idx] = deltas[idx] + gamma * lamb * masks[idx] * gaes[idx + 1]
    return gaes[:-1]


def revert_norm_returns(rewards: Tensor, dones: Tensor, gamma: float=0.99) -> Tensor:
    """
    Parameters:
        rewards: Rewards to discount. Expected shape (..., 1)
        dones: Tensor with termination flag. Expected ints {0, 1} in shape (..., 1)
        gamma: Discount factor.
    """
    discounted_reward = torch.zeros(rewards.shape[1:])
    returns = torch.zeros_like(rewards).float()
    len_returns = returns.shape[0]
    for idx, (reward, done) in enumerate(zip(reversed(rewards), reversed(dones.int()))):
        discounted_reward = reward + gamma * discounted_reward * (1 - done)
        returns[len_returns - idx - 1] = discounted_reward

    returns = (returns - returns.mean(dim=0)) / torch.clamp(returns.std(dim=0), 1e-8)
    return returns
