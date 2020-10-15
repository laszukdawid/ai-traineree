from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from typing import Optional, Sequence, Tuple, Union


class NetworkType(nn.Module):
    def act(self, *args):
        with torch.no_grad():
            self.eval()
            x = self.forward(*args)
            self.train()
            return x


def hidden_init(layer: nn.Module):
    fan_in = layer.weight.data.size()[0]  # type: ignore
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


def layer_init(layer: nn.Module, range_value: Optional[Tuple[float, float]]=None):
    if not (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
        return
    if range_value is not None:
        layer.weight.data.uniform_(*range_value)  # type: ignore

    nn.init.xavier_uniform_(layer.weight)


class QNetwork(NetworkType):
    def __init__(self, state_size: Union[Sequence[int], int], action_size: int, hidden_layers: Sequence[int]):
        super(QNetwork, self).__init__()

        state_size_list = list(state_size) if not isinstance(state_size, int) else [state_size]
        layers_conn = state_size_list + list(hidden_layers) + [action_size]
        layers = [nn.Linear(layers_conn[idx], layers_conn[idx + 1]) for idx in range(len(layers_conn) - 1)]
        self.layers = nn.ModuleList(layers)
        self.reset_parameters()

        self.gate = F.relu

    def reset_parameters(self):
        for layer in self.layers[:-1]:
            layer_init(layer, hidden_init(layer))
        layer_init(self.layers[-1], (-1e-3, 1e-3))

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class QNetwork2D(NetworkType):
    def __init__(self, state_dim: Sequence[int], action_size, hidden_layers: Sequence[int]):
        super(QNetwork2D, self).__init__()

        # state_dim = (num_layers, x_img, y_img)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 4, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(8, 8),
        )

        output_size = reduce(lambda a, b: a*b, self._calculate_output_size(state_dim, self.conv_layers))
        layers_conn = [output_size] + list(hidden_layers) + [action_size]

        fc_layers = [nn.Linear(layers_conn[idx], layers_conn[idx + 1]) for idx in range(len(layers_conn) - 1)]
        self.fc_layers = nn.ModuleList(fc_layers)

        self.reset_parameters()
        self.gate = F.relu
        self.gate_out = F.softmax

    @staticmethod
    def _calculate_output_size(input_dim: Sequence[int], conv_layers):
        test_tensor = torch.zeros((1,) + tuple(input_dim))
        with torch.no_grad():
            out = conv_layers(test_tensor)
        return out.shape

    def reset_parameters(self):
        self.conv_layers.apply(layer_init)
        for layer in self.fc_layers[:-1]:
            layer_init(layer, hidden_init(layer))
        layer_init(self.fc_layers[-1], (-1e-3, 1e-3))

    def forward(self, x):
        x = self.conv_layers(x)

        x = x.view(x.size(0), -1)
        for layer in self.fc_layers[:-1]:
            x = self.gate(layer(x))
        x = self.fc_layers[-1](x)
        return self.gate_out(x, dim=-1)


class FcNet(NetworkType):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: Sequence[int]=(200, 100),
                 gate=F.leaky_relu, gate_out=torch.tanh, last_layer_range=(-3e-3, 3e-3),
                 device: Optional[torch.device]=None,
                 ):
        super(FcNet, self).__init__()

        num_layers = [input_dim] + list(hidden_layers) + [output_dim]
        layers = [nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(num_layers[:-1], num_layers[1:])]

        self.last_layer_range = last_layer_range
        self.layers = nn.ModuleList(layers)
        self.reset_parameters()

        self.gate = gate
        self.gate_out = gate_out

    def reset_parameters(self):
        for layer in self.layers[:-1]:
            layer_init(layer, hidden_init(layer))
        layer_init(self.layers[-1], self.last_layer_range)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.gate(layer(x))
        if self.gate_out is None:
            return self.layers[-1](x)
        return self.gate_out(self.layers[-1](x))


class ActorBody(NetworkType):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: Sequence[int]=(200, 100),
                 gate=F.elu, gate_out=torch.tanh, last_layer_range=(-3e-3, 3e-3)):
        super(ActorBody, self).__init__()

        num_layers = [input_dim] + list(hidden_layers) + [output_dim]
        layers = [nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(num_layers[:-1], num_layers[1:])]

        self.last_layer_range = last_layer_range
        self.layers = nn.ModuleList(layers)
        self.reset_parameters()

        self.gate = gate
        self.gate_out = gate_out

    def reset_parameters(self):
        for layer in self.layers[:-1]:
            layer_init(layer, hidden_init(layer))
        layer_init(self.layers[-1], self.last_layer_range)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.gate(layer(x))
        if self.gate_out is None:
            return self.layers[-1](x)
        return self.gate_out(self.layers[-1](x))


class CriticBody(NetworkType):
    def __init__(self, input_dim: int, action_size: int, hidden_layers: Sequence[int]=(200, 100)):
        super(CriticBody, self).__init__()

        num_layers = [input_dim] + list(hidden_layers) + [1]
        layers = [nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(num_layers[:-1], num_layers[1:])]

        # Injects `actions` into the second layer of the Critic
        layers[1] = nn.Linear(num_layers[1]+action_size, num_layers[2])
        self.layers = nn.ModuleList(layers)
        self.reset_parameters()

        self.gate = F.elu

    def reset_parameters(self):
        for layer in self.layers:
            layer_init(layer, hidden_init(layer))

    def act(self, state, actions):
        with torch.no_grad():
            self.eval()
            x = self.forward(state, actions)
            self.train()
            return x

    def forward(self, x, actions):
        for idx, layer in enumerate(self.layers[:-1]):
            if idx == 1:
                x = self.gate(layer(torch.cat((x, actions.float()), dim=-1)))
            else:
                x = self.gate(layer(x))
        return self.layers[-1](x)


class DoubleCritic(NetworkType):
    def __init__(self, input_dim: int, action_size: int, hidden_layers: Sequence[int]=(200, 100)):
        super(DoubleCritic, self).__init__()
        self.critic_1 = CriticBody(input_dim=input_dim, action_size=action_size, hidden_layers=hidden_layers)
        self.critic_2 = CriticBody(input_dim=input_dim, action_size=action_size, hidden_layers=hidden_layers)

    def reset_parameters(self):
        self.critic_1.reset_parameters()
        self.critic_2.reset_parameters()

    def act(self, states, actions):
        return (self.critic_1.act(states, actions), self.critic_2.act(states, actions))

    def forward(self, state, actions):
        return (self.critic_1(state, actions), self.critic_2(state, actions))


class DuelingNet(NetworkType):
    def __init__(self, state_size: int, action_size: int, hidden_layers: Sequence[int],
                 net_fn: Optional[Callable[..., NetworkType]]=None, net_class: Optional[Type[TNetworkType]]=None,
                 **kwargs
                 ):
        super(DuelingNet, self).__init__()
        device = kwargs.get("device")
        if net_fn is not None:
            self.value_net = net_fn(state_size, 1, hidden_layers=hidden_layers)
            self.advantage_net = net_fn(state_size, action_size, hidden_layers=hidden_layers)
        elif net_class is not None:
            self.value_net = net_class(state_size, 1, hidden_layers=hidden_layers, device=device)
            self.advantage_net = net_class(state_size, action_size, hidden_layers=hidden_layers, device=device)
        else:
            self.value_net = FcNet(state_size, 1, hidden_layers=hidden_layers, gate_out=None, device=device)
            self.advantage_net = FcNet(state_size, action_size, hidden_layers=hidden_layers, gate_out=None, device=device)

    def reset_parameters(self) -> None:
        self.value_net.reset_parameters()
        self.advantage_net.reset_parameters()

    def act(self, x):
        value = self.value_net.act(x).float()
        advantage = self.advantage_net.act(x).float()
        q = value.expand_as(advantage) + (advantage - advantage.mean(1, keepdim=True).expand_as(advantage))
        return q

    def forward(self, x):
        value = self.value_net(x).float()
        advantage = self.advantage_net(x).float()
        q = value.expand_as(advantage) + (advantage - advantage.mean(1, keepdim=True).expand_as(advantage))
        return q

class RainbowNet(NetworkType, nn.Module):
    def __init__(self, state_dim, action_dim, num_atoms, hidden_layers=(200, 200), noisy=False, device=None):
        super(RainbowNet, self).__init__()
        if noisy:
            self.fc_value = NoisyNet(state_dim, num_atoms, hidden_layers=hidden_layers, gate=F.elu)
            self.fc_advantage = NoisyNet(state_dim, action_dim*num_atoms, hidden_layers=hidden_layers, gate=F.elu)
        else:
            self.fc_value = FcNet(state_dim, num_atoms, hidden_layers=hidden_layers, gate_out=None)
            self.fc_advantage = FcNet(state_dim, action_dim*num_atoms, hidden_layers=hidden_layers, gate_out=None)

        self.noisy = noisy
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.to(device=device)

    def reset_noise(self):
        if self.noisy:
            self.fc_value.reset_noise()
            self.fc_advantage.reset_noise()

    def act(self, x, log_prob=False):
        """
        :param log_prob: bool
            Whether to return log(prob) which uses pytorch's function. According to doc it's quicker and more stable
            than taking prob.log().
        """
        with torch.no_grad():
            self.eval()
            value = self.fc_value.act(x).view(-1, 1, self.num_atoms)
            advantage = self.fc_advantage.act(x).view(-1, self.action_dim, self.num_atoms)
            q = value + advantage - advantage.mean(1, keepdim=True)
            out = F.softmax(q, dim=-1) if not log_prob else F.log_softmax(q, dim=-1)  # Doc: It's computationally quicker than log(softmax) and more stable
            self.train()
        return out

    def forward(self, x, log_prob=False):
        """
        :param log_prob: bool
            Whether to return log(prob) which uses pytorch's function. According to doc it's quicker and more stable
            than taking prob.log().
        """
        value = self.fc_value(x).view((-1, 1, self.num_atoms))
        advantage = self.fc_advantage(x).view(-1, self.action_dim, self.num_atoms)
        q = value + advantage - advantage.mean(1, keepdim=True)
        if log_prob:
            return F.log_softmax(q, dim=-1)  # Doc: It's computationally quicker than log(softmax) and more stable
        return F.softmax(q, dim=-1)


class NoisyLayer(nn.Module):
    def __init__(self, in_size: int, out_size: int, sigma: float=0.4, factorised: bool=True):
        """
        A Linear layer with values being pertrubed by the noise while training.

        :param sigma: float
            Used to intiated noise distribution.
        :param factorised: bool
            Whether to use independent Gaussian (False) or Factorised Gaussian (True) noise.
            Suggested [1] for DQN and Duelling nets to use factorised as it's quicker.

        Based on:
        [1] "Noisy Networks for Exploration" by Fortunato et al. (ICLR 2018), https://arxiv.org/abs/1706.10295.
        """
        super(NoisyLayer, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.sigma_0 = sigma
        self.factorised = factorised

        self.weight_mu = nn.Parameter(torch.zeros((out_size, in_size)))
        self.weight_sigma = nn.Parameter(torch.zeros((out_size, in_size)))

        self.bias_mu = nn.Parameter(torch.zeros(out_size))
        self.bias_sigma = nn.Parameter(torch.zeros(out_size))

        self.register_buffer('weight_eps', torch.zeros((out_size, in_size)))
        self.register_buffer('bias_eps', torch.zeros(out_size))

        self.bias_noise = torch.zeros(out_size)
        if factorised:
            self.weight_noise = torch.zeros(in_size)
        else:
            self.weight_noise = torch.zeros(out_size, in_size)

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        weight = self.weight_mu
        bias = self.bias_mu
        if self.training:
            weight = weight.add(self.weight_sigma.mul(self.weight_eps))
            bias = bias.add(self.bias_sigma.mul(self.bias_eps))

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        if self.factorised:
            bound = sqrt(1./self.in_size)
            sigma = self.sigma_0 * bound
        else:
            bound = sqrt(3./self.in_size)
            sigma = 0.017  # Yes, that's correct. [1]

        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(sigma)

        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(sigma)

    def reset_noise(self):
        self.bias_noise.normal_(std=self.sigma_0)
        self.weight_noise.normal_(std=self.sigma_0)

        if self.factorised:
            self.weight_eps.copy_(self.noise_function(self.bias_noise).ger(self.noise_function(self.weight_noise)))
            self.bias_eps.copy_(self.noise_function(self.bias_noise))
        else:
            self.weight_eps.copy_(self.weight_noise.data)
            self.bias_eps.copy_(self.bias_noise.data)

    @staticmethod
    def noise_function(x):
        return x.sign().mul_(x.abs().sqrt())


class NoisyNet(NetworkType):
    def __init__(self, in_size: int, out_size: int, hidden_layers=(100, 100), sigma=0.4,
                 gate=None, gate_out=None, factorised=True, device: Optional[torch.device]=None):
        """
            :param factorised: bool
                Whether to use independent Gaussian (False) or Factorised Gaussian (True) noise.
                Suggested [1] for DQN and Duelling nets to use factorised as it's quicker.
        """
        super(NoisyNet, self).__init__()

        num_layers = [in_size] + list(hidden_layers) + [out_size]
        layers = [NoisyLayer(dim_in, dim_out, sigma=sigma, factorised=factorised) for dim_in, dim_out in zip(num_layers[:-1], num_layers[1:])]
        self.layers = nn.ModuleList(layers)

        self.gate = gate if gate is not None else lambda x: x
        self.gate_out = gate_out if gate_out is not None else lambda x: x
        self.to(device=device)

    def reset_noise(self):
        for layer in self.layers:
            layer.reset_noise()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.gate(layer(x))
        return self.gate_out(self.layers[-1](x))
