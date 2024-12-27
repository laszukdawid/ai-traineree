import copy
from unittest.mock import MagicMock

import mock
import pytest
import torch

from ai_traineree.agents.rainbow import RainbowAgent
from ai_traineree.networks.heads import RainbowNet
from ai_traineree.types import AgentState, BufferState, DataSpace, NetworkState
from ai_traineree.types.experience import Experience
from conftest import deterministic_interactions, fake_step, feed_agent

t_obs_space = DataSpace(dtype="float", shape=(10,), low=-1, high=1)
t_action_space = DataSpace(dtype="int", shape=(4,), low=0, high=4)


def test_rainbow_init_fail_without_state_action_dim():
    # Missing both state and action
    with pytest.raises(TypeError):
        RainbowAgent()

    # Missing action
    with pytest.raises(TypeError):
        RainbowAgent(3)


def test_rainbow_init_default():
    # Assign
    agent = RainbowAgent(t_obs_space, t_action_space)

    # Assert
    assert agent.using_double_q
    assert agent.n_steps > 0
    assert isinstance(agent.net, RainbowNet)
    assert isinstance(agent.target_net, RainbowNet)
    assert agent.obs_space == t_obs_space
    assert agent.action_space == t_action_space


def test_rainbow_seed():
    # Assign
    agent_0 = RainbowAgent(t_obs_space, t_action_space, device="cpu")
    agent_1 = RainbowAgent(t_obs_space, t_action_space, device="cpu")
    agent_2 = copy.deepcopy(agent_1)

    # Act
    # Make sure agents have the same networks
    agent_nets = zip(agent_1.net.value_net.layers, agent_2.net.value_net.layers)
    agent_target_nets = zip(agent_1.target_net.value_net.layers, agent_2.target_net.value_net.layers)
    assert all([sum(sum(l1.weight - l2.weight)) == 0 for l1, l2 in agent_nets])
    assert all([sum(sum(l1.weight - l2.weight)) == 0 for l1, l2 in agent_target_nets])

    agent_0.seed(32167)
    actions_0 = deterministic_interactions(agent_0)
    agent_1.seed(0)
    actions_1 = deterministic_interactions(agent_1)
    agent_2.seed(0)
    actions_2 = deterministic_interactions(agent_2)

    # Assert
    assert any(a0 != a1 for (a0, a1) in zip(actions_0, actions_1))
    for idx, (a1, a2) in enumerate(zip(actions_1, actions_2)):
        assert a1 == a2, f"Action mismatch on position {idx}: {a1} != {a2}"


def test_rainbow_set_loss():
    # Assign
    _os = DataSpace(shape=(1,), dtype="int", low=0, high=1)
    _as = DataSpace(shape=(1,), dtype="int", low=0, high=1)
    agent = RainbowAgent(_os, _as, device="cpu")
    new_loss = 1
    assert str(agent.loss) == str({"loss": float("nan")})  # Check default

    # Act
    agent.loss = new_loss

    # Assert
    assert agent.loss == {"loss": new_loss}


@mock.patch("ai_traineree.agents.rainbow.soft_update")
def test_rainbow_warm_up(mock_soft_update):
    """Until `warm_up` iterations passed there can't be any update."""
    # Assign
    warm_up = 10
    # Update will commence after batch is available and on (iter % update_freq) == 0
    agent = RainbowAgent(t_obs_space, t_action_space, device="cpu", warm_up=warm_up, batch_size=1, update_freq=1)
    agent.learn = MagicMock()

    # Act & assert
    for _ in range(warm_up - 1):
        obs, reward, term, trunc = fake_step((1,))
        agent.step(Experience(obs=obs, reward=reward, done=(term or trunc), next_obs=obs, action=1))
        agent.learn.assert_not_called()
        assert not mock_soft_update.called

    obs, reward, term, trunc = fake_step((1,))
    agent.step(Experience(obs=obs, reward=reward, done=(term or trunc), next_obs=obs, action=1))
    agent.learn.assert_called()
    assert mock_soft_update.called


@mock.patch("ai_traineree.loggers.DataLogger")
def test_rainbow_log_metrics(mock_data_logger):
    # Assign
    agent = RainbowAgent(t_obs_space, t_action_space, device="cpu")
    step = 10
    agent.loss = 1

    # Act
    agent.log_metrics(mock_data_logger, step)

    # Assert
    mock_data_logger.log_value.assert_called_once_with("loss/agent", agent._loss, step)
    mock_data_logger.log_value_dict.assert_not_called()
    mock_data_logger.create_histogram.assert_not_called()


@mock.patch("ai_traineree.loggers.DataLogger")
def test_rainbow_log_metrics_full_log(mock_data_logger):
    # Assign
    agent = RainbowAgent(
        t_obs_space, t_action_space, device="cpu", hidden_layers=(10,)
    )  # Only 2 layers ((I, H) -> (H, O)
    step = 10
    agent.loss = 1

    # Act
    agent.log_metrics(mock_data_logger, step, full_log=True)

    # Assert
    assert agent.dist_probs is None
    mock_data_logger.log_value.assert_called_once_with("loss/agent", agent._loss, step)
    mock_data_logger.log_value_dict.assert_not_called()
    mock_data_logger.add_histogram.assert_not_called()
    assert mock_data_logger.create_histogram.call_count == 4 * 2  # 4x per layer


@mock.patch("ai_traineree.loggers.DataLogger")
def test_rainbow_log_metrics_full_log_dist_prob(mock_data_logger):
    """Acting on a state means that there's a prob dist created for each actions."""
    # Assign
    _os = DataSpace(shape=(1,), dtype="int", low=0, high=1)
    _as = DataSpace(shape=(1,), dtype="int", low=0, high=1)
    agent = RainbowAgent(_os, _as, device="cpu", hidden_layers=(10,))  # Only 2 layers (I, H) -> (H, O)
    step = 10
    agent.loss = 1

    # Act
    experience = Experience(obs=[0])
    agent.act(experience)
    agent.log_metrics(mock_data_logger, step, full_log=True)

    # Assert
    assert agent.dist_probs is not None
    mock_data_logger.log_value.call_count == 2  # 1x loss + 1x dist_prob
    mock_data_logger.log_value_dict.assert_not_called()
    mock_data_logger.add_histogram.call_count == 2  # (_as.high - _as.low + 1)
    assert mock_data_logger.create_histogram.call_count == 4 * 2  # 4x per layer


@mock.patch("torch.save")
def test_rainbow_save_state(mock_torch_save):
    # Assign
    agent = RainbowAgent(t_obs_space, t_action_space, device="cpu")
    fname = "/tmp/my_path_is_better_than_yours"

    # Act
    agent.save_state(fname)

    # Assert
    mock_torch_save.assert_called_once_with(mock.ANY, fname)


@mock.patch("torch.load")
def test_rainbow_load_state(mock_torch_load):
    # Assign
    agent = RainbowAgent(t_obs_space, t_action_space, device="cpu")
    agent.net = MagicMock()
    agent.target_net = MagicMock()
    mock_torch_load.return_value = {
        "config": dict(batch_size=10, n_steps=2),
        "net": [1, 2],
        "target_net": [11, 22],
    }

    # Act
    agent.load_state("/tmp/nah_ah_my_path_is_better")

    # Assert
    agent.net.load_state_dict.assert_called_once_with([1, 2])
    agent.target_net.load_state_dict.assert_called_once_with([11, 22])
    assert agent.batch_size == 10
    assert agent.n_steps == 2
    assert agent._config == dict(batch_size=10, n_steps=2)


def test_rainbow_get_state():
    # Assign
    init_config = {"lr": 0.1, "gamma": 0.6}
    agent = RainbowAgent(t_obs_space, t_action_space, device="cpu", **init_config)

    # Act
    agent_state = agent.get_state()

    # Assert
    assert isinstance(agent_state, AgentState)
    assert agent_state.model == RainbowAgent.model
    assert agent_state.obs_space == t_obs_space
    assert agent_state.action_space == t_action_space
    assert agent_state.config == agent._config
    assert agent_state.config["lr"] == 0.1
    assert agent_state.config["gamma"] == 0.6

    network_state = agent_state.network
    assert isinstance(network_state, NetworkState)
    assert {"net", "target_net"} == set(network_state.net.keys())

    buffer_state = agent_state.buffer
    assert isinstance(buffer_state, BufferState)
    assert buffer_state.type == agent.buffer.type
    assert buffer_state.batch_size == agent.buffer.batch_size
    assert buffer_state.buffer_size == agent.buffer.buffer_size


def test_rainbow_get_state_compare_different_agents():
    # Assign
    agent_1 = RainbowAgent(t_obs_space, t_action_space, device="cpu", n_steps=1)
    agent_2 = RainbowAgent(t_obs_space, t_action_space, device="cpu", n_steps=2)

    # Act
    state_1 = agent_1.get_state()
    state_2 = agent_2.get_state()

    # Assert
    assert state_1 != state_2
    assert state_1.model == state_2.model


def test_rainbow_from_state():
    # Assign
    agent = RainbowAgent(t_obs_space, t_action_space, device="cpu")
    agent_state = agent.get_state()

    # Act
    new_agent = RainbowAgent.from_state(agent_state)

    # Assert
    assert id(agent) != id(new_agent)
    # assert new_agent == agent
    assert isinstance(new_agent, RainbowAgent)
    assert new_agent.hparams == agent.hparams
    assert all([torch.all(x == y) for (x, y) in zip(agent.net.parameters(), new_agent.net.parameters())])
    assert all([torch.all(x == y) for (x, y) in zip(agent.target_net.parameters(), new_agent.target_net.parameters())])
    assert new_agent.buffer == agent.buffer


def test_rainbow_from_state_one_updated():
    # Assign
    agent = RainbowAgent(t_obs_space, t_action_space, device="cpu")
    feed_agent(agent, 2 * agent.batch_size)  # Feed 1
    agent_state = agent.get_state()
    feed_agent(agent, 100)  # Feed 2 - to make different

    # Act
    new_agent = RainbowAgent.from_state(agent_state)

    # Assert
    assert id(agent) != id(new_agent)
    # assert new_agent == agent
    assert isinstance(new_agent, RainbowAgent)
    assert new_agent.hparams == agent.hparams
    assert any([torch.any(x != y) for (x, y) in zip(agent.net.parameters(), new_agent.net.parameters())])
    assert any([torch.any(x != y) for (x, y) in zip(agent.target_net.parameters(), new_agent.target_net.parameters())])
    assert new_agent.buffer != agent.buffer
