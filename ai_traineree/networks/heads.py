"""
Heads are build on Brains.
Like in real life, heads do all the difficult part of receiving stimuli,
being above everything else and not falling apart.
You take brains out and they just do nothng. Lazy.
The most common use case is when one head contains one brain.
But who are we to say what you can and cannot do.
You want two brains and a head within your head? Sure, go crazy.

What we're trying to do here is to keep thing relatively simple.
Unfortunately, not everything can be achieved [citation needed] with a serial
topography and at some point you'll need branching.
Heads are "special" in that each is built on networks/brains and will likely need
some special pipeping when attaching to your agent.
"""
from functools import lru_cache, reduce
from operator import mul
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_traineree.networks import NetworkType, NetworkTypeClass
from ai_traineree.networks.bodies import FcNet, NoisyNet
from ai_traineree.types import FeatureType
from ai_traineree.utils import to_numbers_seq


class NetChainer(NetworkType):
    """Chains nets into a one happy family.

    As it stands it is a wrapper around pytroch.nn.ModuleList.
    The need for wrapper comes from unified API to reset properties.
    """

    def __init__(self, net_classes: List[NetworkTypeClass], **kwargs):
        super(NetChainer, self).__init__()
        self.nets = nn.ModuleList(net_classes)

        self.in_features = self._determin_feature_size(self.nets[0].layers[0], is_in=True)
        self.out_features = self._determin_feature_size(self.nets[-1].layers[-1], is_in=False)

    @staticmethod
    def _determin_feature_size(layer, is_in=True):
        if "Conv" in str(layer):
            return layer.in_channels if is_in else layer.out_channels
        else:
            return layer.in_features if is_in else layer.out_features

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
    def __init__(self, in_features: Sequence[int], action_size: int, body_cls: NetworkTypeClass, **kwargs):
        super(DoubleCritic, self).__init__()
        hidden_layers = kwargs.pop("hidden_layers", (200, 200))
        self.critic_1 = body_cls(
            in_features=in_features, inj_action_size=action_size, hidden_layers=hidden_layers, **kwargs
        )
        self.critic_2 = body_cls(
            in_features=in_features, inj_action_size=action_size, hidden_layers=hidden_layers, **kwargs
        )

    def reset_parameters(self):
        self.critic_1.reset_parameters()
        self.critic_2.reset_parameters()

    def act(self, states, actions):
        return (self.critic_1.act(states, actions), self.critic_2.act(states, actions))

    def forward(self, state, actions):
        return (self.critic_1(state, actions), self.critic_2(state, actions))


class DuelingNet(NetworkType):
    def __init__(
        self,
        in_features: Sequence[int],
        out_features: Sequence[int],
        hidden_layers: Sequence[int],
        net_fn: Optional[Callable[..., NetworkType]] = None,
        net_class: Optional[NetworkTypeClass] = None,
        **kwargs
    ):
        """
        Parameters:
            in_features (tuple of ints): Dimension of the input features.
            out_features (tuple of ints): Dimension of critic's action. Default: (1,).
            hidden_layers (tuple of ints): Shape of the hidden layers.
            net_fn (optional func):
            net_class (optional class)

        Keyword arguments:
            device: Device where to allocate memory. CPU or CUDA. Default CUDA if available.

        """
        super(DuelingNet, self).__init__()
        device = kwargs.get("device")
        # We only care about the leading size, e.g. (4,) -> 4
        if net_fn is not None:
            self.value_net = net_fn(in_features, (1,), hidden_layers=hidden_layers)
            self.advantage_net = net_fn(in_features, out_features, hidden_layers=hidden_layers)
        elif net_class is not None:
            self.value_net = net_class(in_features, (1,), hidden_layers=hidden_layers, device=device)
            self.advantage_net = net_class(in_features, out_features, hidden_layers=hidden_layers, device=device)
        else:
            self.value_net = FcNet(
                in_features, (1,), hidden_layers=hidden_layers, gate_out=nn.Identity(), device=device
            )
            self.advantage_net = FcNet(
                in_features, out_features, hidden_layers=hidden_layers, gate_out=nn.Identity(), device=device
            )

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

    References:
        .. [1] "A Distributional Perspective on Reinforcement Learning" (2017) by M. G. Bellemare, W. Dabney, R. Munos.
            Link: http://arxiv.org/abs/1707.06887

    """

    def __init__(
        self,
        num_atoms: int = 21,
        v_min: float = -20.0,
        v_max: float = 20.0,
        in_features: Optional[FeatureType] = None,
        out_features: Optional[FeatureType] = None,
        hidden_layers: Sequence[int] = (200, 200),
        net: Optional[NetworkType] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Parameters:
            num_atoms: Number of atoms that disceritze the probability distrubition.
            v_min: Minimum (edge) value of the shifted distribution.
            v_max: Maximum (edge) value of the shifted distribution.
            net: (Optional) A network used for estimation. If `net` is proved then `hidden_layers` has no effect.
            obs_space: Size of the observation.
            action_size: Length of the output.
            hidden_layers: Shape of the hidden layers that are fully connected networks.

        *Note* that either `net` or both (`obs_space`, `action_size`) need to be not None.
        If `obs_space` and `action_size` are provided then the default net is created as
        fully connected network with `hidden_layers` size.

        """
        super(CategoricalNet, self).__init__()
        self.device = device
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.z_atoms = torch.linspace(v_min, v_max, num_atoms, device=device)
        self.z_delta = self.z_atoms[1] - self.z_atoms[0]

        if net is not None:
            self.net = net
        elif in_features is not None and out_features is not None:
            assert len(out_features) == 1, "Expecting single dimension for output features"
            _out_features = (out_features[0] * self.num_atoms,)
            self.net = FcNet(in_features, _out_features, hidden_layers=hidden_layers, device=self.device)
        else:
            raise ValueError(
                "CategoricalNet needs to be instantiated either with `net` or (`obs_space` and `action_size`)"
            )

        assert len(self.net.out_features) == 1, "Expecting single dimension for output features"
        self.in_featores = self.net.in_features
        self.out_features = (self.net.out_features[0] // self.num_atoms, self.num_atoms)
        self.to(device=device)

    def reset_paramters(self):
        self.net.reset_parameters()

    def forward(self, *args) -> torch.Tensor:
        """
        Passes *args through the net with proper handling.
        """
        return self.net(*args).view((-1,) + self.out_features)

    @lru_cache(maxsize=5)
    def _offset(self, batch_size, device=None):
        offset = torch.linspace(0, ((batch_size - 1) * self.num_atoms), batch_size, device=self.device)
        return offset.unsqueeze(1).expand(batch_size, self.num_atoms)

    def mean(self, values):
        return (self.z_atoms * values).mean()

    def dist_projection(
        self, rewards: torch.Tensor, masks: torch.Tensor, discount: float, prob_next: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters:
            rewards: Tensor containing rewards that are used as offsets for each distrubitions.
            masks: Tensor indicating whether the iteration is terminal. Usually `masks = 1 - dones`.
            discount: Discounting value for added Q distributional estimate. Typically gamma or gamma^(n_steps).
            prob_next: Probablity estimates based on transitioned (next) states.

        """
        batch_size = rewards.shape[0]
        Tz = rewards + discount * masks * self.z_atoms.view(1, -1)
        assert Tz.shape == (batch_size, self.num_atoms)
        Tz.clamp_(self.v_min, self.v_max - 1e-4)  # In place. Tiny eps required for num stability e.g. ceil(1.00000001)

        b_idx = (Tz - self.v_min) / self.z_delta
        l_idx = b_idx.floor().to(torch.int64)
        u_idx = b_idx.ceil().to(torch.int64)

        # Fix disappearing probability mass when l = b = u (b is int)
        # Checking twice `l_idx == u_idx` is on purpose, since we first want to distribute to the left
        # but in cases we can't go any lower (already on the boundary) we will move them higher.
        l_idx[torch.logical_and(l_idx == u_idx, u_idx > 0)] -= 1
        u_idx[torch.logical_and(l_idx == u_idx, l_idx < self.num_atoms - 1)] += 1

        offset = self._offset(batch_size)
        l_offset_idx = (l_idx + offset).type(torch.int64)
        u_offset_idx = (u_idx + offset).type(torch.int64)

        # Distribute probability of Tz
        m = rewards.new_zeros(batch_size * self.num_atoms)

        # Dealing with indices. *Note* not to forget batches.
        # m[l] = m[l] + p(s[t+n], a*)(u - b)
        m.index_add_(0, l_offset_idx.view(-1), (prob_next * (u_idx.float() - b_idx)).view(-1))
        # m[u] = m[u] + p(s[t+n], a*)(b - l)
        m.index_add_(0, u_offset_idx.view(-1), (prob_next * (b_idx - l_idx.float())).view(-1))

        return m.view(batch_size, self.num_atoms)


class RainbowNet(NetworkType, nn.Module):
    """Rainbow networks combines dueling and categorical networks."""

    def __init__(self, in_features: FeatureType, out_features: FeatureType, **kwargs):
        """
        Parameters
            in_features (tuple of ints): Shape of the input.
            out_features (tuple of ints): Shape of the expected output.

        Keyword arguments:
            hidden_layers (tuple of ints): Shape of fully connected networks. Default: (200, 200).
            num_atoms (int): Number of atoms used in estimating distribution. Default: 21.
            v_min (float): Value distribution minimum (left most) value. Default -10.
            v_max (float): Value distribution maximum (right most) value. Default 10.
            noisy (bool): Whether to use Noisy version of FC networks.
            pre_network_fn (func): A shared network that is used before *value* and *advantage* networks.
            device (None, str or torch.device): Device where to cast the network. Can be assigned with strings, or
                directly passing torch.device type. If `None` then it tries to use CUDA then CPU. Default: None.

        """
        super(RainbowNet, self).__init__()
        self.device = device = kwargs.get("device", None)

        self.pre_network = None
        if "pre_network_fn" in kwargs:
            self.pre_network = kwargs.get("pre_network_fn")(in_features=in_features)
            self.pre_netowrk_params = self.pre_network.parameters()  # Registers pre_network's parameters to this module
            pof = self.pre_network.out_features
            in_features = (pof,) if isinstance(pof, int) else pof

        self.v_min = float(kwargs.get("v_min", -10))
        self.v_max = float(kwargs.get("v_max", 10))
        self.num_atoms = num_atoms = int(kwargs.get("num_atoms", 21))
        self.z_atoms = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=self.device)
        self.z_delta = self.z_atoms[1] - self.z_atoms[0]

        in_size, out_size = reduce(mul, in_features), reduce(mul, out_features)
        hidden_layers = to_numbers_seq(kwargs.get("hidden_layers", (128, 128)))
        self.noisy = kwargs.get("noisy", False)
        if self.noisy:
            self.value_net = NoisyNet((in_size,), out_features=(num_atoms,), hidden_layers=hidden_layers, device=device)
            self.advantage_net = NoisyNet((in_size,), out_size * num_atoms, hidden_layers=hidden_layers, device=device)
        else:
            self.value_net = FcNet(in_features, out_features=(num_atoms,), hidden_layers=hidden_layers, device=device)
            self.advantage_net = FcNet(in_features, (out_size * num_atoms,), hidden_layers=hidden_layers, device=device)

        if self.pre_network is not None:
            pif = self.pre_network.in_features
            self.in_features = (pif,) if isinstance(pif, int) else pif
        else:
            self.in_features = in_features
        self.out_features = out_features
        self.to(device=self.device)

    @lru_cache(maxsize=5)
    def _offset(self, batch_size):
        offset = torch.linspace(0, ((batch_size - 1) * self.num_atoms), batch_size, device=self.device)
        return offset.unsqueeze(1).expand(batch_size, self.num_atoms)

    def reset_noise(self):
        if self.noisy:
            self.value_net.reset_noise()
            self.advantage_net.reset_noise()

    def act(self, x, log_prob=False):
        """
        Parameters:
            log_prob (bool):
                Whether to return log(prob) which uses pytorch's function. According to doc it's quicker and more stable
                than taking prob.log().
        """
        with torch.no_grad():
            self.eval()
            if self.pre_network is not None:
                x = self.pre_network(x)
            value = self.value_net.act(x).view(-1, 1, self.num_atoms)
            advantage = self.advantage_net.act(x).view((-1,) + self.out_features + (self.num_atoms,))
            q = value + advantage - advantage.mean(1, keepdim=True)
            # Doc: It's computationally quicker than log(softmax) and more stable
            out = F.softmax(q, dim=-1) if not log_prob else F.log_softmax(q, dim=-1)
            self.train()
        return out

    def forward(self, x, log_prob=False):
        """
        Parameters:
            log_prob (bool):
                Whether to return log(prob) which uses pytorch's function. According to doc it's quicker and more stable
                than taking prob.log().
        """
        if self.pre_network is not None:
            x = self.pre_network(x)
        value = self.value_net(x).view((-1, 1, self.num_atoms))
        advantage = self.advantage_net(x).view((-1,) + self.out_features + (self.num_atoms,))
        q = value + advantage - advantage.mean(1, keepdim=True)
        if log_prob:
            # Doc: It's computationally quicker than log(softmax) and more stable
            return F.log_softmax(q, dim=-1)
        return F.softmax(q, dim=-1)

    def dist_projection(
        self, rewards: torch.Tensor, masks: torch.Tensor, discount: float, prob_next: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters:
            rewards: Tensor containing rewards that are used as offsets for each distrubitions.
            masks: Tensor indicating whether the iteration is terminal. Usually `masks = 1 - dones`.
            discount: Discounting value for added Q distributional estimate. Typically gamma or gamma^(n_steps).
            prob_next: Probablity estimates based on transitioned (next) states.

        """
        batch_size = rewards.shape[0]
        Tz = rewards + discount * masks * self.z_atoms.view(1, -1)
        assert Tz.shape == (batch_size, self.num_atoms)
        Tz.clamp_(self.v_min, self.v_max)  # in place

        b_idx = (Tz - self.v_min) / self.z_delta
        l_idx = b_idx.floor().to(torch.int64)
        u_idx = b_idx.ceil().to(torch.int64)

        # Fix disappearing probability mass when l = b = u (b is int)
        l_idx[(u_idx > 0) * (l_idx == u_idx)] -= 1
        u_idx[(l_idx < (self.num_atoms - 1)) * (l_idx == u_idx)] += 1

        offset = self._offset(batch_size)
        l_offset_idx = (l_idx + offset).type(torch.int64)
        u_offset_idx = (u_idx + offset).type(torch.int64)

        # Distribute probability of Tz
        m = rewards.new_zeros(batch_size * self.num_atoms)

        # Dealing with indices. *Note* not to forget batches.
        # m[l] = m[l] + p(s[t+n], a*)(u - b)
        m.index_add_(0, l_offset_idx.view(-1), (prob_next * (u_idx.float() - b_idx)).view(-1))
        # m[u] = m[u] + p(s[t+n], a*)(b - l)
        m.index_add_(0, u_offset_idx.view(-1), (prob_next * (b_idx - l_idx.float())).view(-1))

        return m.view(batch_size, self.num_atoms)
