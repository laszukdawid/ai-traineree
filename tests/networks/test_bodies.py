import ai_traineree.networks.bodies as bodies
import torch


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
    in_features, out_features = 8, 2
    net = bodies.FcNet(in_features, out_features)
    test_tensor = torch.randn((1, in_features))

    # Act
    out_tensor = net(test_tensor)

    # Assert
    assert net.in_features == in_features
    assert net.out_features == out_features
    assert out_tensor.shape == (1, out_features)


def test_fc_hidden_layers_none():
    # Assign
    in_features, out_features = 9, 3
    net = bodies.FcNet(in_features, out_features, hidden_layers=None)
    test_tensor = torch.randn((3, in_features))

    # Act
    out_tensor = net(test_tensor)

    # Assert
    assert net.in_features == in_features
    assert net.out_features == out_features
    assert len(net.layers) == 1
    assert out_tensor.shape == (3, out_features)


def test_actor_body():
    """Test that the ActorBody net is the same as FcNet."""
    assert bodies.FcNet == bodies.ActorBody


def test_critic_body_default():
    # Assign
    in_features, action_dim = 10, 4
    test_tensor = torch.randn((1, in_features))
    test_action = torch.randn((1, action_dim))
    net = bodies.CriticBody(in_features, action_dim)
    
    # Act
    out_tensor = net(test_tensor, test_action)

    # Assert
    assert net.in_features == in_features
    assert net.out_features == 1
    assert out_tensor.shape == (1, 1)
    assert net.layers[1].in_features == net.layers[0].out_features + action_dim


if __name__ == "__main__":
    test_scalenet()