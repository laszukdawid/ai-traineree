from functools import reduce
from math import sqrt
from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_traineree.networks import NetworkType
from ai_traineree.types import FeatureType


def hidden_init(layer: nn.Module):
    fan_in = layer.weight.data.size()[0]  # type: ignore
    lim = 1.0 / sqrt(fan_in)
    return (-lim, lim)


def layer_init(layer: nn.Module, range_value: Optional[Tuple[float, float]] = None, remove_mean=True):
    if not (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
        return
    if range_value is not None:
        layer.weight.data.uniform_(*range_value)  # type: ignore
    if remove_mean:
        layer.weight.data -= layer.weight.data.mean()

    nn.init.xavier_uniform_(layer.weight)


class ScaleNet(NetworkType):
    def __init__(self, scale: Union[float, int]) -> None:
        super(ScaleNet, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class ConvNet(NetworkType):
    def __init__(self, input_dim: Sequence[int], **kwargs):
        """Convolution Network.

        Constructs a layered network over torch.nn.Conv2D. Number of layers is set based on `hidden_layers` argument.
        To update other arguments, e.g. kernel_size or bias, pass either a single value or a tuple of the same
        length as `hidden_layers`.

        Quick reminder from the PyTorch doc (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).

        Keyword Arguments:
            in_channels (int): Number of channels in the input image
            hidden_layers (tuple of ints): Number of channels in each hidden layer
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
            padding_mode (string, optional): 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
            bias (bool, optional): If True, adds a learnable bias to the output. Default: True

        Examples:
            >>> config = {"hidden_layers": (300, 200, 100), "kernel_size": 6, "gate": F.relu}
            >>> net = ConvNet(input_dim=(10, 10, 3), **config)
            >>> config = {"hidden_layers": (64, 32, 64), "kernel_size": (3, 4, 3), padding: 2, "gate": F.relu}
            >>> net = ConvNet(input_dim=(20, 10, 1), **config)

        """
        super(ConvNet, self).__init__()

        # input_dim = (num_layers, x_img, y_img, channels)
        hidden_layers = kwargs.get("hidden_layers", (20, 20))
        num_layers = [input_dim[0]] + list(hidden_layers)

        gate = kwargs.get("gate", nn.ReLU)
        max_pool_sizes = self._expand_to_seq(kwargs.get("max_pool_size", 2), len(hidden_layers))
        kernel_sizes = self._expand_to_seq(kwargs.get("kernel_size", 3), len(hidden_layers))
        strides = self._expand_to_seq(kwargs.get("stride", 1), len(hidden_layers))
        paddings = self._expand_to_seq(kwargs.get("padding", 0), len(hidden_layers))
        dilations = self._expand_to_seq(kwargs.get("dilation", 1), len(hidden_layers))
        biases = self._expand_to_seq(kwargs.get("bias", True), len(hidden_layers))

        layers = []
        for layer_idx in range(len(hidden_layers)):
            layers.append(
                nn.Conv2d(
                    num_layers[layer_idx],
                    num_layers[layer_idx + 1],
                    kernel_size=kernel_sizes[layer_idx],
                    stride=strides[layer_idx],
                    padding=paddings[layer_idx],
                    dilation=dilations[layer_idx],
                    bias=biases[layer_idx],
                )
            )

            if max_pool_sizes[layer_idx] > 1:
                layers.append(nn.MaxPool2d(max_pool_sizes[layer_idx]))

            if gate is not None:
                layers.append(gate())

        self.layers = nn.ModuleList(layers)
        self.reset_parameters()

        self.input_dim = input_dim
        self.device = kwargs.get("device")
        self.to(self.device)

    @staticmethod
    def _expand_to_seq(o: Union[Any, Sequence[Any]], size) -> Sequence[Any]:
        return o if isinstance(o, Sequence) else (o,) * size

    @property
    def output_size(self):
        return reduce(lambda a, b: a * b, self._calculate_output_size(self.input_dim, self.layers))

    @torch.no_grad()
    def _calculate_output_size(self, input_dim: Sequence[int], layers) -> Sequence[int]:
        test_tensor = torch.zeros((1,) + tuple(input_dim)).to(self.device)
        out = self.forward(test_tensor)
        return out.shape

    def reset_parameters(self):
        self.layers.apply(layer_init)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FcNet(NetworkType):
    """
    For the activation layer we use tanh by default which was observed to be much better, e.g. compared to ReLU,
    for policy networks [1]. The last gate, however, might be changed depending on the actual task.


    References
        .. [1] "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study"
            by M. Andrychowicz et al. (2020). Link: https://arxiv.org/abs/2006.05990

    """

    def __init__(
        self,
        in_features: FeatureType,
        out_features: FeatureType,
        hidden_layers: Optional[Sequence[int]] = (200, 100),
        last_layer_range=(-3e-4, 3e-4),
        bias: bool = True,
        **kwargs,
    ):
        """Fully Connected network with default APIs.

        Parameters:
            in_features (sequence of ints): Shape of the input.
            out_features (sequence of ints): Shape of the output.
            hidden_layers: Shape of the hidden layers. If None, then the output is directly computed from the input.
            last_layer_range: The range for the uniform distribution that initiates the last layer.

        Keyword arguments:
            gate (optional torch.nn.layer): Activation function for each layer, expect the last. Default: torch.tanh.
            gate_out (optional torch.nn.layer): Activation function after the last layer. Default: Identity layer.
            device (torch.devce or str): Device where to allocate memory. CPU or CUDA.


        """
        super(FcNet, self).__init__()
        assert len(in_features) == 1, "Expected only one dimension"
        assert len(out_features) == 1, "Expected only one dimension"

        self.in_features = tuple(in_features)
        self.out_features = tuple(out_features)
        num_layers = tuple(hidden_layers) if hidden_layers is not None else tuple()
        num_layers = self.in_features + num_layers + self.out_features
        layers = [nn.Linear(dim_in, dim_out, bias=bias) for dim_in, dim_out in zip(num_layers[:-1], num_layers[1:])]

        self.last_layer_range = last_layer_range
        self.layers = nn.ModuleList(layers)
        self.reset_parameters()

        self.gate = kwargs.get("gate", torch.tanh)
        self.gate_out = kwargs.get("gate_out", nn.Identity())
        self.to(kwargs.get("device"))

    def reset_parameters(self):
        for layer in self.layers[:-1]:
            layer_init(layer, hidden_init(layer), remove_mean=True)
        layer_init(self.layers[-1], self.last_layer_range, remove_mean=True)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.gate(layer(x))
        return self.gate_out(self.layers[-1](x))


###
# In most cases, the default ActorBody can be associated with a fully connected network.
# The alias is only for convinience and, hopefully, better understanding of some algorithms.
###
ActorBody = FcNet


class CriticBody(NetworkType):
    """Extension of the FcNet which includes actions.

    Mainly used to estimate the state-action value function in actor-critic agents.
    Actions are included (by default) in the first hidden layer (changeable).

    Since the main purpose for this is value function estimation the output is a single value.
    """

    def __init__(
        self,
        in_features: FeatureType,
        inj_action_size: int,
        out_features: FeatureType = (1,),
        hidden_layers: Optional[Sequence[int]] = (100, 100),
        inj_actions_layer: int = 1,
        **kwargs,
    ):
        """
        Parameters:
            in_features (tuple of ints): Dimension of the input features.
            inj_action_size (int): Dimension of the action vector that is injected into `inj_action_layer`.
            out_features (tuple of ints): Dimension of critic's action. Default: (1,).
            hidden_layers (tuple of ints): Shape of the hidden layers. Default: (100, 100).
            inj_action_layer (int): An index for the layer that will have `actions` injected as an additional input.
                By default that's a first hidden layer, i.e. (state) -> (out + actions) -> (out) ... -> (output).
                Default: 1.

        Keyword arguments:
            bias (bool): Whether to include bias in network's architecture. Default: True.
            gate (callable): Activation function for each layer, expect the last. Default: Identity layer.
            gate_out (callable): Activation function after the last layer. Default: Identity layer.
            device: Device where to allocate memory. CPU or CUDA. Default CUDA if available.

        """
        super().__init__()

        self.in_features = tuple(in_features)
        self.out_features = tuple(out_features)
        num_layers = tuple(hidden_layers) if hidden_layers is not None else tuple()
        num_layers = self.in_features + num_layers + self.out_features
        self.actions_layer = inj_actions_layer
        if not (0 <= inj_actions_layer < len(num_layers)):
            raise ValueError("Action layer needs to be within the network")

        bias = bool(kwargs.get("bias", True))
        layers = []
        for in_idx in range(len(num_layers) - 1):
            in_dim, out_dim = num_layers[in_idx], num_layers[in_idx + 1]
            if in_idx == inj_actions_layer:  # Injects `actions` into specified (default: 2nd) layer of the Critic
                in_dim += inj_action_size
            layers.append(nn.Linear(in_dim, out_dim, bias=bias))

        self.layers = nn.ModuleList(layers)
        self.reset_parameters()

        self.gate = kwargs.get("gate", nn.Identity())
        self.gate_out = kwargs.get("gate_out", nn.Identity())
        self.to(kwargs.get("device"))

    def reset_parameters(self):
        for layer in self.layers:
            layer_init(layer, hidden_init(layer))

    def forward(self, x, actions):
        for idx, layer in enumerate(self.layers):
            if idx == self.actions_layer:
                x = layer(torch.cat((x, actions.float()), dim=-1))
            else:
                x = layer(x)

            if idx < len(self.layers) - 1:
                x = self.gate(x)
            else:
                x = self.gate_out(x)
        return x


class NoisyLayer(nn.Module):
    def __init__(
        self, in_features: FeatureType, out_features: FeatureType, sigma: float = 0.4, factorised: bool = True
    ):
        """
        A linear layer with added noise perturbations in training as described in [1].
        For a fully connected network of NoisyLayers see :class:`.NoisyNet`.

        Parameters:
            in_features (tuple ints): Dimension of the input.
            out_features (tuple ints): Dimension of the output.
            sigma (float): Used to intiated noise distribution. Default: 0.4.
            factorised: Whether to use independent Gaussian (False) or Factorised Gaussian (True) noise.
                Suggested [1] for DQN and Duelling nets to use factorised as it's quicker.

        References:
            .. [1] "Noisy Networks for Exploration" by Fortunato et al. (ICLR 2018), https://arxiv.org/abs/1706.10295.

        """
        super(NoisyLayer, self).__init__()
        assert len(in_features) == 1, "Expected only one dimension"
        assert len(out_features) == 1, "Expected only one dimension"

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma
        self.factorised = factorised

        self.weight_mu = nn.Parameter(torch.zeros((out_features[0], in_features[0])))
        self.weight_sigma = nn.Parameter(torch.zeros((out_features[0], in_features[0])))

        self.bias_mu = nn.Parameter(torch.zeros(out_features[0]))
        self.bias_sigma = nn.Parameter(torch.zeros(out_features[0]))

        self.register_buffer("weight_eps", torch.zeros((out_features[0], in_features[0])))
        self.register_buffer("bias_eps", torch.zeros(out_features[0]))

        self.bias_noise = torch.zeros(out_features[0])
        if factorised:
            self.weight_noise = torch.zeros(in_features[0])
        else:
            self.weight_noise = torch.zeros(out_features[0], in_features[0])

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x) -> torch.Tensor:
        weight = self.weight_mu
        bias = self.bias_mu
        if self.training:
            weight = weight.add(self.weight_sigma.mul(self.weight_eps))
            bias = bias.add(self.bias_sigma.mul(self.bias_eps))

        return F.linear(x, weight, bias)

    def reset_parameters(self) -> None:
        if self.factorised:
            bound = sqrt(1.0 / self.in_features[0])
            sigma = self.sigma_0 * bound
        else:
            bound = sqrt(3.0 / self.in_features[0])
            sigma = 0.017  # Yes, that's correct. [1]

        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(sigma)

        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(sigma)

    def reset_noise(self):
        self.bias_noise.normal_(std=self.sigma_0)
        self.weight_noise.normal_(std=self.sigma_0)

        if self.factorised:
            # eps_i = ~P(b_size),  eps_j = ~P(w_size)
            # eps_b = f(eps_i)
            # eps_w = f(eps_i) x f(eps_j)
            f_weight_eps = self.noise_function(self.weight_noise)
            f_bias_eps = self.noise_function(self.bias_noise)
            self.weight_eps.copy_(f_bias_eps.outer(f_weight_eps))
            self.bias_eps.copy_(f_bias_eps)
        else:
            self.weight_eps.copy_(self.weight_noise.data)
            self.bias_eps.copy_(self.bias_noise.data)

    @staticmethod
    def noise_function(x):
        return x.sign().mul_(x.abs().sqrt())


class NoisyNet(NetworkType):
    def __init__(
        self,
        in_features: FeatureType,
        out_features: FeatureType,
        hidden_layers: Optional[Sequence[int]] = (100, 100),
        sigma=0.4,
        factorised=True,
        **kwargs,
    ):
        """
        Parameters:
            in_features (tuple ints): Dimension of the input.
            out_features (tuple ints): Dimension of the output.
            hidden_layers (sequence ints): Sizes of latent layers. Size of sequence denotes number of hidden layers and
                values of the sequence are nodes per layer. If None is passed then the input goes straight to output.
                Default: (100, 100).
            sigma (float): Variance value for generating noise in noisy layers. Default: 0.4 per layer.
            factorised (bool): Whether to use independent Gaussian (False) or Factorised Gaussian (True) noise.
                Suggested [1] for DQN and Duelling nets to use factorised as it's quicker.

        Keyword arguments:
            gate (callable): Function to apply after each layer pass. For the best performance it is suggested
                to use non-linear functions such as tanh. Default: tanh.
            gate_out (callable): Function to apply on network's exit. Default: identity.
            device (str or torch.device): Whether and where to cast the network. Default is CUDA if available else cpu.

        References:
            .. [1] "Noisy Networks for Exploration" by Fortunato et al. (ICLR 2018), https://arxiv.org/abs/1706.10295.

        """
        super(NoisyNet, self).__init__()
        assert len(in_features) == 1, "Expected only one dimension"
        assert len(out_features) == 1, "Expected only one dimension"

        self.in_features = in_features
        self.out_features = out_features
        num_layers = list(hidden_layers) if hidden_layers is not None else []
        num_layers = [self.in_features[0]] + num_layers + [self.out_features[0]]
        layers = [
            NoisyLayer((dim_in,), (dim_out,), sigma=sigma, factorised=factorised)
            for dim_in, dim_out in zip(num_layers[:-1], num_layers[1:])
        ]
        self.layers = nn.ModuleList(layers)

        self.gate = kwargs.get("gate", torch.tanh)
        self.gate_out = kwargs.get("gate_out", nn.Identity())
        if not callable(self.gate) or not callable(self.gate_out):
            raise ValueError("Passed gate or gate_out is no callable and cannot be used as a function")

        self.to(device=kwargs.get("device", None))

    def reset_noise(self) -> None:
        for layer in self.layers:
            layer.reset_noise()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.gate(layer(x))
        return self.gate_out(self.layers[-1](x))
