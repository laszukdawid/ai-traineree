import torch
import torch.nn as nn

from typing import List


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)  # type: ignore


def hard_update(target: nn.Module, source: nn.Module):
    """Updates one network based on another."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)  # type: ignore


def to_np(t):
    return t.cpu().detach().numpy()


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def revert_norm_returns(rewards, dones, gamma=0.99) -> torch.Tensor:
    discounted_reward = 0
    returns: List[float] = []
    for reward, done in zip(reversed(rewards), reversed(dones)):
        discounted_reward = reward + gamma * discounted_reward * (1 - done)
        returns.insert(0, discounted_reward)

    t_returns = torch.tensor(returns)
    t_returns = (t_returns - t_returns.mean()) / (t_returns.std() + 1e-8)
    return t_returns
