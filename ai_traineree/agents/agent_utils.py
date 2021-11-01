import torch
import torch.nn as nn
from torch import Tensor

EPS = 1e-7


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)  # type: ignore


def hard_update(target: nn.Module, source: nn.Module):
    """Updates one network based on another."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)  # type: ignore


def compute_gae(rewards: Tensor, dones: Tensor, values: Tensor, next_value: Tensor, gamma=0.99, lamb=0.9) -> Tensor:
    """Uses General Advantage Estimator to compute... general advantage estimation."""
    _tmp_values = torch.cat((values, next_value[None, ...]))
    masks = 1 - dones.int()
    gaes = torch.zeros_like(_tmp_values)
    deltas = rewards + gamma * _tmp_values[1:] * masks - _tmp_values[:-1]
    gamma_lambda = gamma * lamb
    for idx in reversed(range(len(rewards))):
        gaes[idx] = deltas[idx] + gamma_lambda * masks[idx] * gaes[idx + 1]
    return gaes[:-1]


def normalize(t: Tensor, dim: int = 0) -> Tensor:
    """Returns normalized (zero 0 and std 1) tensor along specified axis (default: 0)."""
    if dim == 0:
        # Special case since by default it reduces on dim 0 and it should be faster.
        return (t - t.mean(dim=dim)) / torch.clamp(t.std(dim=dim), EPS)
    else:
        return (t - t.mean(dim=dim, keepdim=True)) / torch.clamp(t.std(dim=dim, keepdim=True), EPS)


def revert_norm_returns(rewards: Tensor, dones: Tensor, gamma: float = 0.99) -> Tensor:
    """
    Parameters:
        rewards: Rewards to discount. Expected shape (..., 1)
        dones: Tensor with termination flag. Expected ints {0, 1} in shape (..., 1)
        gamma: Discount factor.
    """
    discounted_reward = torch.zeros(rewards.shape[1:], dtype=rewards.dtype, device=rewards.device)
    returns = torch.zeros_like(rewards).float()
    len_returns = returns.shape[0]
    for idx, (reward, done) in enumerate(zip(reversed(rewards), reversed(dones.int()))):
        discounted_reward = reward + gamma * discounted_reward * (1 - done)
        returns[len_returns - idx - 1] = discounted_reward

    return normalize(returns, dim=0)
