import torch
import torch.nn as nn

from ai_traineree.utils import to_tensor
from torch import Tensor
from typing import List


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
    masks = 1 - dones
    gaes = torch.zeros_like(_tmp_values)
    deltas = rewards + gamma * _tmp_values[1:] * masks - _tmp_values[:-1]
    for idx in reversed(range(len(rewards))):
        gaes[idx] = deltas[idx] + gamma * lamb * masks[idx] * gaes[idx + 1]
    return gaes[:-1]


def revert_norm_returns(rewards, dones, gamma=0.99, device=None) -> torch.Tensor:
    discounted_reward = 0
    returns: List[torch.Tensor] = []
    for reward, done in zip(reversed(rewards), reversed(dones)):
        discounted_reward = reward + gamma * discounted_reward * (1 - done)
        returns.insert(0, discounted_reward)

    t_returns = to_tensor(returns).to(device)
    t_returns = (t_returns - t_returns.mean()) / torch.clamp(t_returns.std(), 1e-8)
    return t_returns
