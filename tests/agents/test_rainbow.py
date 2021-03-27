import copy
import mock
import pytest

from ai_traineree.networks.heads import RainbowNet
from ai_traineree.agents.rainbow import RainbowAgent as Agent
from conftest import deterministic_interactions, fake_step
from unittest.mock import MagicMock


def RainbowAgent(*args, **kwargs):
    """Monkey patch to make sure all RainbowAgents are running on cpu."""
    kwargs['device'] = 'cpu'
    return Agent(*args, **kwargs)


def test_rainbow_init_fail_without_state_action_dim():
    # Missing both state and action
    with pytest.raises(TypeError):
        RainbowAgent()

    # Missing action
    with pytest.raises(TypeError):
        RainbowAgent(3)


def test_rainbow_init_default():
    # Assign
    input_shape, output_shape = (10,), (2,)
    agent = RainbowAgent(input_shape, output_shape)

    # Assert
    assert agent.using_double_q
    assert agent.n_steps > 0
    assert isinstance(agent.net, RainbowNet)
    assert isinstance(agent.target_net, RainbowNet)
    assert agent.state_size == input_shape[0]
    assert agent.action_size == output_shape[0]


def test_rainbow_seed():
    # Assign
    agent_0 = RainbowAgent(4, 4, device='cpu')
    agent_1 = RainbowAgent(4, 4, device='cpu')
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
    agent = RainbowAgent(1, 1)
    new_loss = 1
    assert agent.loss == {'loss': float('inf')} # Check default

    # Act
    agent.loss = new_loss

    # Assert
    assert agent.loss == {'loss': new_loss}


@mock.patch("ai_traineree.agents.rainbow.soft_update")
def test_rainbow_warm_up(mock_soft_update):
    """Until `warm_up` iterations passed there can't be any update."""
    # Assign
    warm_up = 10
    # Update will commence after batch is available and on (iter % update_freq) == 0
    agent = RainbowAgent(1, 1, warm_up=warm_up, batch_size=1, update_freq=1)
    agent.learn = MagicMock()

    # Act & assert
    for _ in range(warm_up - 1):
        state, reward, done = fake_step((1,))
        agent.step(state, 1, reward, state, done)
        agent.learn.assert_not_called()
        assert not mock_soft_update.called

    state, reward, done = fake_step((1,))
    agent.step(state, 1, reward, state, done)
    agent.learn.assert_called()
    assert mock_soft_update.called


@mock.patch("ai_traineree.loggers.DataLogger")
def test_rainbow_log_metrics(mock_data_logger):
    # Assign
    agent = RainbowAgent(1, 1)
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
    agent = RainbowAgent(1, 1, hidden_layers=(10,))  # Only 2 layers ((I, H) -> (H, O)
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
    agent = RainbowAgent(1, 1, hidden_layers=(10,))  # Only 2 layers ((I, H) -> (H, O)
    step = 10
    agent.loss = 1

    # Act
    agent.act([0])
    agent.log_metrics(mock_data_logger, step, full_log=True)

    # Assert
    assert agent.dist_probs is not None
    mock_data_logger.log_value.call_count == 2  # 1x loss + 1x dist_prob
    mock_data_logger.log_value_dict.assert_not_called()
    mock_data_logger.add_histogram.assert_called_once()
    assert mock_data_logger.create_histogram.call_count == 4 * 2  # 4x per layer


@mock.patch("torch.save")
def test_rainbow_save_state(mock_torch_save):
    # Assign
    agent = RainbowAgent(1, 1)
    fname = "/tmp/my_path_is_better_than_yours"

    # Act
    agent.save_state(fname)

    # Assert
    print(fname)
    mock_torch_save.assert_called_once_with(mock.ANY, fname)


@mock.patch("torch.load")
def test_rainbow_load_state(mock_torch_load):
    # Assign
    agent = RainbowAgent(2, 2)
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


if __name__ == "__main__":
    test_rainbow_set_loss()
