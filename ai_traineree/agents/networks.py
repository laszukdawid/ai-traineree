import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hidden_init(layer: nn.Module):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def layer_init(layer: nn.Module, range_value: Tuple[float, float]):
    layer.weight.data.uniform_(*range_value)

class ActorBody(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[64,64]):
        super(ActorBody, self).__init__()

        num_layers = [input_dim] + list(hidden_layers) + [output_dim]
        layers = [ nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(num_layers[:-1], num_layers[1:]) ]

        self.layers = nn.ModuleList(layers)
        self.reset_parameters()

        self.gate = F.relu
        self.gate_out = torch.tanh

    def reset_parameters(self):
        for layer in self.layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))
        layer_init(self.layers[-1], (-3e-3, 3e-3))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.gate(layer(x))
        return self.gate_out(self.layers[-1](x))

class CriticBody(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_layers=[64,64]):
        super(CriticBody, self).__init__()

        num_layers = [input_dim] + list(hidden_layers) + [1]
        layers = [nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(num_layers[:-1], num_layers[1:])]

        # Injects `actions` into the second layer of the Critic
        layers[1] = nn.Linear(num_layers[1]+action_dim, num_layers[2])
        self.layers = nn.ModuleList(layers)


        self.gate = F.leaky_relu
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))
        layer_init(self.layers[-1], (-3e-3, 3e-3))

    def forward(self, x, actions):
        for idx, layer in enumerate(self.layers[:-1]):
            if idx == 1:
                x = self.gate(layer(torch.cat((x, actions), dim=1)))
            else:
                x = self.gate(layer(x))
        return self.layers[-1](x)
