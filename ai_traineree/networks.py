import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, Tuple

def hidden_init(layer: nn.Module):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def layer_init(layer: nn.Module, range_value: Tuple[float, float]):
    layer.weight.data.uniform_(*range_value)

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, hidden_layer: Sequence[int]):
        super(QNetwork, self).__init__()

        layers_conn = [state_size] + list(hidden_layer) + [action_size]
        layers = [nn.Linear(layers_conn[idx], layers_conn[idx+1]) for idx in range(len(layers_conn)-1)]
        self.layers = nn.ModuleList(layers) 

        self.gate = F.relu

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x