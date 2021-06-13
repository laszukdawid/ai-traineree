import copy

import pytest
import torch

from ai_traineree.agents.ppo import PPOAgent
from ai_traineree.types.state import AgentState, BufferState, NetworkState
from conftest import deterministic_interactions


def test_ppo_seed():
    # Assign
    agent_0 = PPOAgent(4, 2, device='cpu')  # Reference
    agent_1 = PPOAgent(4, 2, device='cpu')
    agent_2 = copy.deepcopy(agent_1)

    # Act
    # Make sure agents have the same networks
    assert any([sum(sum(l1.weight - l2.weight)) != 0 for l1, l2 in zip(agent_0.actor.layers, agent_1.actor.layers)])
    assert all([sum(sum(l1.weight - l2.weight)) == 0 for l1, l2 in zip(agent_1.actor.layers, agent_2.actor.layers)])
    assert all([sum(sum(l1.weight - l2.weight)) == 0 for l1, l2 in zip(agent_1.critic.layers, agent_2.critic.layers)])

    agent_0.seed(32167)
    actions_0 = deterministic_interactions(agent_0)
    agent_1.seed(0)
    actions_1 = deterministic_interactions(agent_1)
    agent_2.seed(0)
    actions_2 = deterministic_interactions(agent_2)

    # Assert
    # First we check that there's definitely more than one type of action
    assert actions_1[0] != actions_1[1]
    assert actions_2[0] != actions_2[1]

    # All generated actions need to identical
    assert any(a0 != a1 for (a0, a1) in zip(actions_0, actions_1))
    for idx, (a1, a2) in enumerate(zip(actions_1, actions_2)):
        assert a1 == pytest.approx(a2, 1e-4), f"Action mismatch on position {idx}: {a1} != {a2}"


def test_ppo_get_state():
    # Assign
    obs_size, action_size = 3, 4
    init_config = {'actor_lr': 0.1, 'gamma': 0.6}
    agent = PPOAgent(obs_size, action_size, device='cpu', **init_config)

    # Act
    agent_state = agent.get_state()

    # Assert
    assert isinstance(agent_state, AgentState)
    assert agent_state.model == PPOAgent.name
    assert agent_state.obs_space == obs_size
    assert agent_state.action_space == action_size
    assert agent_state.config == agent._config
    assert agent_state.config['actor_lr'] == 0.1
    assert agent_state.config['gamma'] == 0.6

    network_state = agent_state.network
    assert isinstance(network_state, NetworkState)
    assert {'actor', 'critic', 'policy'} == set(network_state.net.keys())

    buffer_state = agent_state.buffer
    assert isinstance(buffer_state, BufferState)
    assert buffer_state.type == agent.buffer.type
    assert buffer_state.batch_size == agent.buffer.batch_size
    assert buffer_state.buffer_size == agent.buffer.buffer_size


def test_ppo_get_state_compare_different_agents():
    # Assign
    obs_size, action_size = 3, 2
    agent_1 = PPOAgent(obs_size, action_size, device='cpu', n_steps=1)
    agent_2 = PPOAgent(obs_size, action_size, device='cpu', n_steps=2)

    # Act
    state_1 = agent_1.get_state()
    state_2 = agent_2.get_state()

    # Assert
    assert state_1 != state_2
    assert state_1.model == state_2.model


def test_ppo_from_state():
    # Assign
    obs_size, action_size = 10, 3
    agent = PPOAgent(obs_size, action_size)
    agent_state = agent.get_state()

    # Act
    new_agent = PPOAgent.from_state(agent_state)

    # Assert
    assert id(agent) != id(new_agent)
    # assert new_agent == agent
    assert isinstance(new_agent, PPOAgent)
    assert new_agent.hparams == agent.hparams
    assert all([torch.all(x == y) for (x, y) in zip(agent.policy.parameters(), new_agent.policy.parameters())])
    assert all([torch.all(x == y) for (x, y) in zip(agent.actor.parameters(), new_agent.actor.parameters())])
    assert all([torch.all(x == y) for (x, y) in zip(agent.critic.parameters(), new_agent.critic.parameters())])
    assert new_agent.buffer == agent.buffer


def test_ppo_from_state_network_state_none():
    # Assign
    obs_size, action_size = 10, 3
    agent = PPOAgent(obs_size, action_size)
    agent_state = agent.get_state()
    agent_state.network = None

    # Act
    new_agent = PPOAgent.from_state(agent_state)

    # Assert
    assert id(agent) != id(new_agent)
    # assert new_agent == agent
    assert isinstance(new_agent, PPOAgent)
    assert new_agent.hparams == agent.hparams
    assert new_agent.buffer == agent.buffer


def test_ppo_from_state_buffer_state_none():
    # Assign
    obs_size, action_size = 10, 3
    agent = PPOAgent(obs_size, action_size)
    agent_state = agent.get_state()
    agent_state.buffer = None

    # Act
    new_agent = PPOAgent.from_state(agent_state)

    # Assert
    assert id(agent) != id(new_agent)
    # assert new_agent == agent
    assert isinstance(new_agent, PPOAgent)
    assert new_agent.hparams == agent.hparams
    assert all([torch.all(x == y) for (x, y) in zip(agent.policy.parameters(), new_agent.policy.parameters())])
    assert all([torch.all(x == y) for (x, y) in zip(agent.actor.parameters(), new_agent.actor.parameters())])
    assert all([torch.all(x == y) for (x, y) in zip(agent.critic.parameters(), new_agent.critic.parameters())])


def test_ppo_from_state_one_updated():
    # Assign
    obs_size, action_size = 10, 3
    agent = PPOAgent(obs_size, action_size)
    deterministic_interactions(agent, num_iters=100)
    agent_state = agent.get_state()
    deterministic_interactions(agent, num_iters=400)

    # Act
    new_agent = PPOAgent.from_state(agent_state)

    # Assert
    assert id(agent) != id(new_agent)
    # assert new_agent == agent
    assert isinstance(new_agent, PPOAgent)
    # assert any([torch.any(x != y) for (x, y) in zip(agent.policy.parameters(), new_agent.policy.parameters())])
    assert any([torch.any(x != y) for (x, y) in zip(agent.actor.parameters(), new_agent.actor.parameters())])
    assert any([torch.any(x != y) for (x, y) in zip(agent.critic.parameters(), new_agent.critic.parameters())])
    assert new_agent.buffer != agent.buffer
