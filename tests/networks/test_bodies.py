import math

import pytest
import torch
import torch.nn as nn

import ai_traineree.networks.bodies as bodies


def test_scalenet():
    # Assign
    test_tensor = torch.arange(1, 100).view(3, 11, 3)
    net = bodies.ScaleNet(0.2)

    # Act
    out_tensor = net(test_tensor)

    # Assert
    assert (test_tensor * 0.2 - out_tensor).abs().sum() == 0
    assert test_tensor.shape == out_tensor.shape


def test_fc_default():
    # Assign
    in_features, out_features = (8,), (2,)
    net = bodies.FcNet(in_features, out_features)
    test_tensor = torch.randn((1,) + in_features)

    # Act
    out_tensor = net(test_tensor)

    # Assert
    assert net.in_features == in_features
    assert net.out_features == out_features
    assert out_tensor.shape == (1,) + out_features
    assert net.gate is not None
    assert isinstance(net.gate, nn.ReLU)
    assert isinstance(net.gate_out, nn.Identity)


def test_fc_hidden_layers_none():
    # Assign
    in_features, out_features = (9,), (3,)
    net = bodies.FcNet(in_features, out_features, hidden_layers=None)
    test_tensor = torch.randn((3,) + in_features)

    # Act
    out_tensor = net(test_tensor)

    # Assert
    assert net.in_features == in_features
    assert net.out_features == out_features
    assert len(net.layers) == 1
    assert out_tensor.shape == (3,) + out_features


def test_fc_forward():
    # Assign
    in_features = (4,)
    out_features = (2,)
    hidden_layers = (10, 10)
    net = bodies.FcNet(in_features=in_features, out_features=out_features, hidden_layers=hidden_layers)
    test_tensor = torch.rand(in_features)

    # Act
    out_implicit = net(test_tensor)
    out_direct = net.forward(test_tensor)

    # Assert
    assert torch.all((out_implicit - out_direct) == 0)
    assert out_implicit.shape == out_features


def test_actor_body():
    """Test that the ActorBody net is the same as FcNet."""
    assert bodies.FcNet == bodies.ActorBody


def test_critic_body_default():
    # Assign
    in_features, action_size = (10,), 4
    test_tensor = torch.randn((1,) + in_features)
    test_action = torch.randn((1, action_size))
    net = bodies.CriticBody(in_features, action_size)

    # Act
    out_tensor = net(test_tensor, test_action)

    # Assert
    assert net.in_features == in_features
    assert net.out_features == (1,)
    assert out_tensor.shape == (1, 1)
    assert net.layers[1].in_features == int(net.layers[0].out_features) + action_size
    assert isinstance(net.gate, nn.Identity)
    assert isinstance(net.gate_out, nn.Identity)


def test_critic_forward():
    in_features = (4,)
    action_size = 2
    net = bodies.CriticBody(in_features=in_features, inj_action_size=action_size, hidden_layers=(20, 20))
    test_tensor = torch.rand(in_features)
    test_action = torch.rand((action_size,))

    # Act
    out = net(test_tensor, test_action)

    # Assert
    assert out.shape == (1,)


def test_critic_forward_out_features():
    in_features = (4,)
    out_features = (4,)
    action_size = 2
    net = bodies.CriticBody(
        in_features=in_features, inj_action_size=action_size, out_features=out_features, hidden_layers=(20, 20)
    )
    test_tensor = torch.rand(in_features)
    test_action = torch.rand((action_size,))

    # Act
    out = net(test_tensor, test_action)

    # Assert
    assert out.shape == out_features


def test_critic_action_layer_outside():
    hidden_layers = (20, 20)
    # 4 layers in total = input + 2 hidden layers + output
    for inj_actions_layer in [-1, 4]:
        with pytest.raises(ValueError):
            bodies.CriticBody(
                in_features=(3,), inj_action_size=4, inj_actions_layer=inj_actions_layer, hidden_layers=hidden_layers
            )


def test_noisy_layer_default():
    in_features, out_features = (4,), (2,)
    layer = bodies.NoisyLayer(in_features, out_features)

    # Assert
    assert layer.sigma_0 == 0.4, "Default sigma is 0.4"
    assert layer.factorised, "By default factorised is enabled"
    assert layer.in_features == in_features
    assert layer.out_features == out_features
    assert layer.weight_noise.shape == in_features
    assert layer.bias_noise.shape == out_features


def test_noisy_layer_not_factorised():
    in_features, out_features = (4,), (2,)
    layer = bodies.NoisyLayer(in_features, out_features, factorised=False)

    # Assert
    assert not layer.factorised, "When not factorised, it should be not factorised. Right?"
    assert layer.sigma_0 == 0.4, "Default sigma is 0.4"
    assert layer.in_features == in_features
    assert layer.out_features == out_features
    assert layer.weight_noise.shape == (out_features[0], in_features[0])
    assert layer.bias_noise.shape == out_features


def test_noisy_layer_forward():
    "By default, layer is in train mode which adds noise on each iteration"
    in_features, out_features = (4,), (2,)
    test_tensor = torch.rand(in_features)
    layer = bodies.NoisyLayer(in_features, out_features)
    # layer.train()

    # Act
    out_implicit = layer(test_tensor)
    out_explicit = layer.forward(test_tensor)

    # Assert
    assert torch.eq(out_implicit, out_explicit).all(), "Values of implicit and explicit need to be the same"
    assert out_implicit.shape == out_features, "Out tensor should be of out_features shape"


def test_noisy_layer_forward_cmp_train_eval():
    # Assign
    in_features, out_features = (4,), (2,)
    test_tensor = torch.rand(in_features)
    layer = bodies.NoisyLayer(in_features, out_features)

    # Act
    layer.train()
    out_train = layer(test_tensor)
    layer.eval()
    out_eval = layer(test_tensor)

    # Assert
    assert torch.ne(out_train, out_eval).any(), "Train and eval modes should produce different tensor due to noise"


def test_noisy_layer_reset_parameters():
    # Assign
    in_features, out_features = (4,), (2,)
    sigma = 0.2
    layer = bodies.NoisyLayer(in_features, out_features, sigma=sigma)

    default_weight_mu = layer.weight_mu.clone().detach()
    default_weight_sigma = layer.weight_sigma.clone().detach()
    default_bias_mu = layer.bias_mu.clone().detach()
    default_bias_sigma = layer.bias_sigma.clone().detach()

    # Act
    layer.reset_parameters()
    weight_mu = layer.weight_mu.clone().detach()
    weight_sigma = layer.weight_sigma.clone().detach()
    bias_mu = layer.bias_mu.clone().detach()
    bias_sigma = layer.bias_sigma.clone().detach()

    # Assert
    assert torch.ne(default_weight_mu, weight_mu).any(), "Weights mu should be uniform sampled"
    assert torch.ne(default_bias_mu, bias_mu).any(), "Bias mu should be uniform sampled"
    assert torch.eq(default_weight_sigma, weight_sigma).any(), "Weights sigma should be predefined"
    assert torch.eq(default_bias_sigma, bias_sigma).any(), "Bias sigma should be predefined"

    assert layer.factorised, "Default factorised is True"
    assert layer.sigma_0 == sigma, "Sigma for noise generation"
    expected_sigma = sigma / math.sqrt(in_features[0])
    assert torch.isclose(bias_sigma, torch.full(bias_sigma.shape, expected_sigma)).all()
    assert torch.isclose(weight_sigma, torch.full(weight_sigma.shape, expected_sigma)).all()


def test_noisy_layer_reset_parameters_not_factorised():
    # Assign
    in_features, out_features = (4,), (2,)
    sigma = 0.2
    layer = bodies.NoisyLayer(in_features, out_features, sigma=sigma, factorised=False)

    default_weight_mu = layer.weight_mu.clone().detach()
    default_weight_sigma = layer.weight_sigma.clone().detach()
    default_bias_mu = layer.bias_mu.clone().detach()
    default_bias_sigma = layer.bias_sigma.clone().detach()

    # Act
    layer.reset_parameters()
    weight_mu = layer.weight_mu.clone().detach()
    weight_sigma = layer.weight_sigma.clone().detach()
    bias_mu = layer.bias_mu.clone().detach()
    bias_sigma = layer.bias_sigma.clone().detach()

    # Assert
    assert torch.ne(default_weight_mu, weight_mu).any(), "Weights mu should be uniform sampled"
    assert torch.ne(default_bias_mu, bias_mu).any(), "Bias mu should be uniform sampled"
    assert torch.eq(default_weight_sigma, weight_sigma).any(), "Weights sigma should be predefined"
    assert torch.eq(default_bias_sigma, bias_sigma).any(), "Bias sigma should be predefined"

    assert not layer.factorised, "Default factorised is True"
    assert layer.sigma_0 == sigma, "When not factorised we use passed sigma"
    assert torch.all(bias_sigma == 0.017), "When not factorised, default is exactly 0.017"
    assert torch.all(weight_sigma == 0.017), "When not factorised, default is exactly 0.017"


def test_noisy_net_default():
    # Assign
    in_features, out_features = (9,), (5,)
    net = bodies.NoisyNet(in_features=in_features, out_features=out_features)

    # Assert
    assert net.in_features == in_features
    assert net.out_features == out_features
    assert net.gate == torch.tanh
    assert isinstance(net.gate_out, nn.Identity)
    assert len(net.layers) == 3, "in -> h1 -> h2 -> out that's 4 concept layers but 3 implemented"


def test_noisy_net_incorrect_init():
    # Wrong input dim
    with pytest.raises(AssertionError):
        bodies.NoisyNet(in_features=(2, 3), out_features=(2,))

    # Wrong output dim
    with pytest.raises(AssertionError):
        bodies.NoisyNet(in_features=(2,), out_features=(2, 3))

    # Wrong gate function
    with pytest.raises(ValueError):
        bodies.NoisyNet(in_features=(3,), out_features=(3,), gate="gate")

    # Wrong gate function
    with pytest.raises(ValueError):
        bodies.NoisyNet(in_features=(3,), out_features=(3,), gate_out="gate_out")


def test_noisy_net_no_hidden_layers():
    # Assign
    in_features, out_features = (2,), (10,)
    net = bodies.NoisyNet(in_features=in_features, out_features=out_features, hidden_layers=None)
    test_tensor = torch.rand(in_features)

    # Act
    out_tensor = net(test_tensor)

    # Assert
    assert len(net.layers) == 1, "Direct layer from input to output"
    assert out_tensor.shape == out_features


def test_noisy_net_forward():
    # Assign
    in_features, out_features = (9,), (5,)
    net = bodies.NoisyNet(in_features=in_features, out_features=out_features)
    test_tensor = torch.rand(in_features)

    # Act
    out_implicit = net(test_tensor)
    out_explicit = net(test_tensor)

    # Assert
    assert out_explicit.shape == out_features
    assert torch.eq(out_implicit, out_explicit).all()


def test_noisy_net_reset_noise():
    # Assign
    in_features, out_features = (9,), (5,)
    hidden_layers = (10, 10)
    net = bodies.NoisyNet(in_features, out_features, hidden_layers=hidden_layers)

    w_noises, b_noises = [], []
    for layer in net.layers:
        w_noises.append(layer.weight_eps.clone().detach())
        b_noises.append(layer.bias_eps.clone().detach())

    # Act
    net.reset_noise()

    # Assert
    num_layers = 1 + len(hidden_layers)
    assert len(net.layers) == num_layers, "Make sure the number of layers is correct"
    for idx, layer in enumerate(net.layers):
        assert torch.ne(w_noises[idx], layer.weight_eps).all()
        assert torch.ne(b_noises[idx], layer.bias_eps).all()


def test_noisy_net_reset_parameters():
    # Assign
    in_features, out_features = (9,), (5,)
    hidden_layers = (10, 10)
    net = bodies.NoisyNet(in_features, out_features, hidden_layers=hidden_layers)

    # Only checking mu since it proves what's needed and sigma maybe be the same
    w_mu, b_mu = [], []
    for layer in net.layers:
        w_mu.append(layer.weight_mu.clone().detach())
        b_mu.append(layer.bias_mu.clone().detach())

    # Act
    net.reset_parameters()

    # Assert
    num_layers = 1 + len(hidden_layers)
    assert len(net.layers) == num_layers, "Make sure the number of layers is correct"
    for idx, layer in enumerate(net.layers):
        assert torch.ne(w_mu[idx], layer.weight_mu).all()
        assert torch.ne(b_mu[idx], layer.bias_mu).all()


if __name__ == "__main__":
    test_scalenet()
