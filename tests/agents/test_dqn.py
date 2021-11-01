import copy

import torch

from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.types import AgentState, BufferState, NetworkState
from ai_traineree.types.dataspace import DataSpace
from conftest import deterministic_interactions, feed_agent

t_obs_space = DataSpace(dtype="float", shape=(4,), low=-2, high=2)
t_action_space = DataSpace(dtype="int", shape=(4,), low=0, high=4)


def test_dqn_seed():
    # Assign
    agent_0 = DQNAgent(t_obs_space, t_action_space, device="cpu")  # Reference
    agent_1 = DQNAgent(t_obs_space, t_action_space, device="cpu")
    agent_2 = copy.deepcopy(agent_1)

    # Act
    # Make sure agents have the same networks
    agent_nets = zip(agent_1.net.value_net.layers, agent_2.net.value_net.layers)
    agent_target_nets = zip(agent_1.target_net.value_net.layers, agent_2.target_net.value_net.layers)
    assert all([sum(torch.sum(l1.weight - l2.weight, 0)) == 0 for l1, l2 in agent_nets])
    assert all([sum(torch.sum(l1.weight - l2.weight, 0)) == 0 for l1, l2 in agent_target_nets])

    agent_0.seed(32167)
    actions_0 = deterministic_interactions(agent_0)
    agent_1.seed(0)
    actions_1 = deterministic_interactions(agent_1)
    agent_2.seed(0)
    actions_2 = deterministic_interactions(agent_2)

    # Assert
    assert any(a0 != a1 for (a0, a1) in zip(actions_0, actions_1))
    # All generated actions need to identical
    for idx, (a1, a2) in enumerate(zip(actions_1, actions_2)):
        assert a1 == a2, f"Action mismatch on position {idx}: {a1} != {a2}"


def test_dqn_get_state():
    # Assign
    init_config = {"lr": 0.1, "gamma": 0.6}
    agent = DQNAgent(t_obs_space, t_action_space, device="cpu", **init_config)

    # Act
    agent_state = agent.get_state()

    # Assert
    assert isinstance(agent_state, AgentState)
    assert agent_state.model == DQNAgent.model
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


def test_dqn_get_state_compare_different_agents():
    # Assign
    agent_1 = DQNAgent(t_obs_space, t_action_space, device="cpu", n_steps=1)
    agent_2 = DQNAgent(t_obs_space, t_action_space, device="cpu", n_steps=2)

    # Act
    state_1 = agent_1.get_state()
    state_2 = agent_2.get_state()

    # Assert
    assert state_1 != state_2
    assert state_1.model == state_2.model


def test_dqn_from_state():
    # Assign
    agent = DQNAgent(t_obs_space, t_action_space)
    agent_state = agent.get_state()

    # Act
    new_agent = DQNAgent.from_state(agent_state)

    # Assert
    assert id(agent) != id(new_agent)
    # assert new_agent == agent
    assert isinstance(new_agent, DQNAgent)
    assert new_agent.hparams == agent.hparams
    assert all([torch.all(x == y) for (x, y) in zip(agent.net.parameters(), new_agent.net.parameters())])
    assert all([torch.all(x == y) for (x, y) in zip(agent.target_net.parameters(), new_agent.target_net.parameters())])
    assert new_agent.buffer == agent.buffer


def test_dqn_from_state_network_state_none():
    # Assign
    agent = DQNAgent(t_obs_space, t_action_space)
    agent_state = agent.get_state()
    agent_state.network = None

    # Act
    new_agent = DQNAgent.from_state(agent_state)

    # Assert
    assert id(agent) != id(new_agent)
    # assert new_agent == agent
    assert isinstance(new_agent, DQNAgent)
    assert new_agent.hparams == agent.hparams
    assert new_agent.buffer == agent.buffer


def test_dqn_from_state_buffer_state_none():
    # Assign
    agent = DQNAgent(t_obs_space, t_action_space)
    agent_state = agent.get_state()
    agent_state.buffer = None

    # Act
    new_agent = DQNAgent.from_state(agent_state)

    # Assert
    assert id(agent) != id(new_agent)
    # assert new_agent == agent
    assert isinstance(new_agent, DQNAgent)
    assert new_agent.hparams == agent.hparams
    assert all([torch.all(x == y) for (x, y) in zip(agent.net.parameters(), new_agent.net.parameters())])
    assert all([torch.all(x == y) for (x, y) in zip(agent.target_net.parameters(), new_agent.target_net.parameters())])


def test_dqn_from_state_one_updated():
    # Assign
    agent = DQNAgent(t_obs_space, t_action_space)
    feed_agent(agent, 2 * agent.batch_size)  # Feed 1
    agent_state = agent.get_state()
    feed_agent(agent, 100)  # Feed 2 - to make different

    # Act
    new_agent = DQNAgent.from_state(agent_state)

    # Assert
    assert id(agent) != id(new_agent)
    # assert new_agent == agent
    assert isinstance(new_agent, DQNAgent)
    assert new_agent.hparams == agent.hparams
    assert any([torch.any(x != y) for (x, y) in zip(agent.net.parameters(), new_agent.net.parameters())])
    assert any([torch.any(x != y) for (x, y) in zip(agent.target_net.parameters(), new_agent.target_net.parameters())])
    assert new_agent.buffer != agent.buffer
