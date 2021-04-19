import pytest

from ai_traineree.agents.agent_factory import AgentFactory
from ai_traineree.agents.ddpg import DDPGAgent
from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.agents.ppo import PPOAgent
from ai_traineree.types.state import AgentState


def test_agent_factory_agent_from_state_wrong_state():
    # Assign
    state = AgentState(
        model="WrongModel", state_space=4, action_space=4,
        config={}, network=None, buffer=None,
    )

    with pytest.raises(ValueError):
        AgentFactory.from_state(state)


def test_agent_factory_dqn_agent_from_state_network_buffer_none():
    # Assign
    state_size, action_size = 10, 5
    agent = DQNAgent(state_size, action_size, device="cpu")
    state = agent.get_state()
    state.network = None
    state.buffer = None

    # Act
    new_agent = AgentFactory.from_state(state)

    # Assert
    assert id(new_agent) != id(agent)
    assert new_agent.name == DQNAgent.name
    assert new_agent.hparams == agent.hparams


def test_agent_factory_dqn_agent_from_state():
    # Assign
    state_size, action_size = 10, 5
    agent = DQNAgent(state_size, action_size, device="cpu")
    state = agent.get_state()

    # Act
    new_agent = AgentFactory.from_state(state)

    # Assert
    assert id(new_agent) != id(agent)
    assert new_agent == agent
    assert new_agent.name == DQNAgent.name
    assert new_agent.hparams == agent.hparams
    assert new_agent.buffer == agent.buffer


def test_agent_factory_ppo_agent_from_state():
    # Assign
    state_size, action_size = 10, 5
    agent = PPOAgent(state_size, action_size, device="cpu")
    state = agent.get_state()

    # Act
    new_agent = AgentFactory.from_state(state)

    # Assert
    assert id(new_agent) != id(agent)
    assert new_agent == agent
    assert new_agent.name == PPOAgent.name
    assert new_agent.hparams == agent.hparams
    assert new_agent.buffer == agent.buffer


def test_agent_factory_ppo_agent_from_state_network_buffer_none():
    # Assign
    state_size, action_size = 10, 5
    agent = PPOAgent(state_size, action_size, device="cpu")
    state = agent.get_state()
    state.network = None
    state.buffer = None

    # Act
    new_agent = AgentFactory.from_state(state)

    # Assert
    assert id(new_agent) != id(agent)
    assert new_agent.name == PPOAgent.name
    assert new_agent.hparams == agent.hparams


def test_agent_factory_ddpg_agent_from_state():
    # Assign
    state_size, action_size = 10, 5
    agent = DDPGAgent(state_size, action_size, device="cpu")
    state = agent.get_state()

    # Act
    new_agent = AgentFactory.from_state(state)

    # Assert
    assert id(new_agent) != id(agent)
    assert new_agent.name == DDPGAgent.name
    assert new_agent == agent
    assert new_agent.hparams == agent.hparams
    assert new_agent.buffer == agent.buffer


def test_agent_factory_ddpg_agent_from_state_network_buffer_none():
    # Assign
    state_size, action_size = 10, 5
    agent = DDPGAgent(state_size, action_size, device="cpu")
    state = agent.get_state()
    state.network = None
    state.buffer = None

    # Act
    new_agent = AgentFactory.from_state(state)

    # Assert
    assert id(new_agent) != id(agent)
    assert new_agent.name == DDPGAgent.name
    assert new_agent.hparams == agent.hparams
