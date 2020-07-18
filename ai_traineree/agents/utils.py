import torch.nn as nn


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)  # type: ignore


def hard_update(target: nn.Module, source: nn.Module):
    """Updates one network based on another."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)  # type: ignore


def to_np(t):
    return t.cpu().detach().numpy()
