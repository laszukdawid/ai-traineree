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
                 gate=F.elu, gate_out=torch.tanh, last_layer_range=(-3e-3, 3e-3)):
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
    def __init__(self, state_size: int, action_size: int, hidden_layers: Sequence[int], precompute_net: Optional[NetworkType]=None):
        super(DuelingNet, self).__init__()
        self.precompute_net = precompute_net
        self.value_net = FcNet(state_size, 1, hidden_layers=hidden_layers, gate_out=None)
        self.advantage_net = FcNet(state_size, action_size, hidden_layers=hidden_layers, gate_out=None)

    def reset_parameters(self) -> None:
        if self.precompute_net and hasattr(self.precompute_net, "reset_parameters"):
            self.precompute_net.reset_parameters()
        self.value_net.reset_parameters()
        self.advantage_net.reset_parameters()

    def forward(self, x):
        if self.precompute_net:
            x = self.precompute_net(x)
        value = self.value_net(x)
        advantange = self.advantage_net(x)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        return q
