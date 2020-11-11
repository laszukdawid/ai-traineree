import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from typing import Callable, Optional, List, Sequence
from ai_traineree.networks import NetworkType, NetworkTypeClass
from ai_traineree.networks.bodies import FcNet, NoisyNet


class NetChainer(NetworkType):
    def __init__(self, net_classes: List[NetworkTypeClass], **kwargs):
        super(NetChainer, self).__init__()
        self.nets = nn.ModuleList(net_classes)

    def reset_parameters(self):
        for net in self.nets:
            if hasattr(net, "reset_parameters"):
                net.reset_parameters()

    def reset_noise(self):
        for net in self.nets:
            if hasattr(net, "reset_noise"):
                net.reset_noise()

    def forward(self, x):
        return reduce(lambda x, net: net(x), self.nets, x)


class DoubleCritic(NetworkType):
    def __init__(self, input_dim: int, action_size: int, critic_body_cls: NetworkTypeClass, **kwargs):
        super(DoubleCritic, self).__init__()
        hidden_layers = kwargs.get("hidden_layers", (200, 200))
        self.critic_1 = critic_body_cls(input_dim=input_dim, action_size=action_size, hidden_layers=hidden_layers)
        self.critic_2 = critic_body_cls(input_dim=input_dim, action_size=action_size, hidden_layers=hidden_layers)

    def reset_parameters(self):
        self.critic_1.reset_parameters()
        self.critic_2.reset_parameters()

    def act(self, states, actions):
        return (self.critic_1.act(states, actions), self.critic_2.act(states, actions))

    def forward(self, state, actions):
        return (self.critic_1(state, actions), self.critic_2(state, actions))


class DuelingNet(NetworkType):
    def __init__(self, state_size: int, action_size: int, hidden_layers: Sequence[int],
                 net_fn: Optional[Callable[..., NetworkType]]=None,
                 net_class: Optional[NetworkTypeClass]=None,
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


class CategoricalNet(NetworkType):
    """
    Computes discrete probability distribution for the state-action Q function.

    CategoricalNet [1] learns significantly different compared to other nets here.
    For this reason it won't be suitable for simple replacement in most (current) agents.
    Please check the Agent whether it supports. 

    The algorithm is used in the RainbowNet but not this particular net.

    [1] "A Distributional Perspective on Reinforcement Learning" (2017) by M. G. Bellemare, W. Dabney, R. Munos.
        Link: http://arxiv.org/abs/1707.06887
    """
    def __init__(self, state_size: int, action_size: int, 
                 n_atoms: int=21, v_min: float=-10., v_max: float=10.,
                 hidden_layers: Sequence[int]=(200, 200),
                 net: Optional[NetworkType]=None,
                 device: Optional[torch.device]=None,
                 ):
        super(CategoricalNet, self).__init__()
        self.action_size = action_size
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.z_atoms = torch.linspace(v_min, v_max, n_atoms, device=device)
        self.z_delta = self.z_atoms[1] - self.z_atoms[0]
        self.net = net if net is not None else NoisyNet(state_size, action_size*n_atoms, hidden_layers=hidden_layers, device=device)
        self.to(device=device)

    def reset_paramters(self):
        self.net.reset_parameters()

    def forward(self, x, log_prob=False) -> torch.Tensor:
        """
        :param log_prob: bool
            Whether to return log(prob) which uses pytorch's function. According to doc it's quicker and more stable
            than taking prob.log().
        """
        return self.net(x).view((-1, self.action_size, self.num_atoms))


class RainbowNet(NetworkType, nn.Module):
    def __init__(self, state_dim, action_dim, num_atoms, hidden_layers=(200, 200), noisy=False, device=None):
        super(RainbowNet, self).__init__()
        if noisy:
            self.fc_value = NoisyNet(state_dim, num_atoms, hidden_layers=hidden_layers)
            self.fc_advantage = NoisyNet(state_dim, action_dim*num_atoms, hidden_layers=hidden_layers)
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
            # Doc: It's computationally quicker than log(softmax) and more stable
            out = F.softmax(q, dim=-1) if not log_prob else F.log_softmax(q, dim=-1)
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
            # Doc: It's computationally quicker than log(softmax) and more stable
            return F.log_softmax(q, dim=-1)
        return F.softmax(q, dim=-1)
