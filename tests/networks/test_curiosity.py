import torch

from aitraineree.networks.curiosity import IntrinsicCuriosityModule


def test_icm_forward():
    obs_shape = (4,)
    action_size = 2
    batch_size = 8

    icm = IntrinsicCuriosityModule(obs_shape, action_size, device="cpu")
    obs = torch.randn(batch_size, *obs_shape)
    next_obs = torch.randn(batch_size, *obs_shape)
    actions = torch.randn(batch_size, action_size)

    phi_next, pred_phi_next, pred_actions = icm(obs, next_obs, actions)

    assert phi_next.shape == (batch_size, icm.feature_dim)
    assert pred_phi_next.shape == (batch_size, icm.feature_dim)
    assert pred_actions.shape == (batch_size, action_size)


def test_icm_intrinsic_reward():
    obs_shape = (4,)
    action_size = 2
    batch_size = 8

    icm = IntrinsicCuriosityModule(obs_shape, action_size, device="cpu")
    obs = torch.randn(batch_size, *obs_shape)
    next_obs = torch.randn(batch_size, *obs_shape)
    actions = torch.randn(batch_size, action_size)

    reward = icm.intrinsic_reward(obs, next_obs, actions)
    assert reward.shape == (batch_size, 1)
    assert (reward >= 0).all()


def test_icm_compute_loss():
    obs_shape = (4,)
    action_size = 2
    batch_size = 8

    icm = IntrinsicCuriosityModule(obs_shape, action_size, device="cpu")
    obs = torch.randn(batch_size, *obs_shape)
    next_obs = torch.randn(batch_size, *obs_shape)
    actions = torch.randn(batch_size, action_size)

    loss, forward_loss, inverse_loss = icm.compute_loss(obs, next_obs, actions)
    assert loss.shape == ()
    assert isinstance(forward_loss, float)
    assert isinstance(inverse_loss, float)

    loss.backward()
    for param in icm.parameters():
        assert param.grad is not None


def test_icm_loss_decreases():
    obs_shape = (4,)
    action_size = 2
    batch_size = 32

    icm = IntrinsicCuriosityModule(obs_shape, action_size, device="cpu")
    optimizer = torch.optim.Adam(icm.parameters(), lr=1e-3)

    obs = torch.randn(batch_size, *obs_shape)
    next_obs = obs + 0.1 * torch.randn(batch_size, *obs_shape)
    actions = torch.randn(batch_size, action_size)

    initial_loss, _, _ = icm.compute_loss(obs, next_obs, actions)
    initial_loss_val = initial_loss.item()

    for _ in range(50):
        optimizer.zero_grad()
        loss, _, _ = icm.compute_loss(obs, next_obs, actions)
        loss.backward()
        optimizer.step()

    final_loss, _, _ = icm.compute_loss(obs, next_obs, actions)
    assert final_loss.item() < initial_loss_val
