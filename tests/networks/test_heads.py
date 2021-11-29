import torch

import ai_traineree.networks.heads as head
from ai_traineree.networks.bodies import FcNet


def test_rainbownet_default():
    # Assign
    in_features, out_features = (10,), (2,)
    num_atoms = 11
    net = head.RainbowNet(in_features, out_features, num_atoms=11)
    test_tensor = torch.randn((1,) + in_features)

    # Act
    out_tensor = net(test_tensor)

    # Assert
    assert net.in_features == in_features
    assert net.out_features == out_features
    assert out_tensor.shape == (test_tensor.shape[0],) + out_features + (num_atoms,)
    assert all(["Linear" in str(layer) for layer in net.value_net.layers])
    assert all(["Linear" in str(layer) for layer in net.advantage_net.layers])
    assert net.pre_network is None


def test_rainbownet_prenetwork():
    # Assign
    in_features, out_features = (10,), (2,)
    num_atoms = 21
    pre_net_out_features = (4,)

    def pre_network_fn(in_features):
        return FcNet(in_features, pre_net_out_features)

    net = head.RainbowNet(in_features, out_features, num_atoms=num_atoms, pre_network_fn=pre_network_fn)
    test_tensor = torch.randn((1,) + in_features)

    # Act
    out_tensor = net(test_tensor)

    # Assert
    assert net.in_features == in_features
    assert net.out_features == out_features
    assert out_tensor.shape == (test_tensor.shape[0],) + out_features + (num_atoms,)
    assert all(["Linear" in str(layer) for layer in net.value_net.layers])
    assert all(["Linear" in str(layer) for layer in net.advantage_net.layers])
    assert all(["Linear" in str(layer) for layer in net.pre_network.layers])
    assert net.value_net.layers[0].in_features == pre_net_out_features[0]
    assert net.advantage_net.layers[0].in_features == pre_net_out_features[0]


def test_rainbownet_input_state_tuple():
    # Assign
    in_features = (4,)
    out_features = (2,)
    num_atoms = 21
    net = head.RainbowNet(in_features, out_features, num_atoms=num_atoms)
    test_tensor = torch.randn((1, in_features[0]))

    # Act
    out_tensor = net(test_tensor)

    # Assert
    assert out_tensor.shape == (1,) + out_features + (num_atoms,)


if __name__ == "__main__":
    test_rainbownet_prenetwork()
