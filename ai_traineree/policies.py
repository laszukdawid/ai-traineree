import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class StochasticActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers, actor=None, critic=None):
        super(StochasticActorCritic, self).__init__()
        self.actor = actor
        self.critic = critic

        self.std = nn.Parameter(torch.rand(action_size)*1e-1)
        self.actor_params = list(self.actor.parameters()) + [self.std]
        self.critic_params = list(self.critic.parameters())

    def forward(self, x):
        # action_mu = self.actor(x.detach())
        action_mu = self.actor(x)
        value = self.critic(x)
        # value = self.critic(x, action_mu.detach())
        # value = self.critic(x.detach())
        # dist = Normal(action_mu, F.softmax(self.std))
        dist = Normal(action_mu, F.relu(self.std))
        return dist, value


class GaussianPolicy(nn.Module):
    def __init__(self, size):
        super(GaussianPolicy, self).__init__()
        self.std = nn.Parameter(torch.rand(size)*1)

    def forward(self, x):
        """Returns distribution"""
        return Normal(x, F.relu(self.std))


class DeterministicPolicy(nn.Module):
    def __init__(self, size):
        super(DeterministicPolicy, self).__init__()

    def formward(self, x):
        return x
