import pytest

from ai_traineree.agents.agent_factory import AgentFactory
from ai_traineree.agents.ddpg import DDPGAgent
from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.agents.ppo import PPOAgent
from ai_traineree.agents.sac import SACAgent
from ai_traineree.types.dataspace import DataSpace
from ai_traineree.types.state import AgentState


def test_agent_factory_agent_from_state_wrong_state():
    # Assign
    state = AgentState(
        model="WrongModel",
        obs_space=4,
        action_space=4,
        config={},
        network=None,
        buffer=None,
    )

    with pytest.raises(ValueError):
        AgentFactory.from_state(state)


def test_agent_factory_dqn_agent_from_state_network_buffer_none():
    # Assign
    obs_space = DataSpace(shape=(5,), dtype="float", low=0, high=2)
    action_space = DataSpace(shape=(1,), dtype="int", low=0, high=5)
    agent = DQNAgent(obs_space, action_space, device="cpu")
    state = agent.get_state()
    state.network = None
    state.buffer = None

    # Act
    new_agent = AgentFactory.from_state(state)

    # Assert
    assert id(new_agent) != id(agent)
    assert new_agent.model == DQNAgent.model
    assert new_agent.hparams == agent.hparams


def test_agent_factory_dqn_agent_from_state():
    # Assign
    obs_space = DataSpace(shape=(10,), dtype="float")
    action_space = DataSpace(shape=(1,), dtype="int", low=1, high=5)
    agent = DQNAgent(obs_space, action_space, device="cpu")
    state = agent.get_state()

    # Act
    new_agent = AgentFactory.from_state(state)

    # Assert
    assert id(new_agent) != id(agent)
    assert new_agent == agent
    assert new_agent.model == DQNAgent.model
    assert new_agent.hparams == agent.hparams
    assert new_agent.buffer == agent.buffer


def test_agent_factory_ppo_agent_from_state():
    # Assign
    obs_space = DataSpace(dtype="float", shape=(10,))
    action_space = DataSpace(dtype="float", shape=(5,))
    agent = PPOAgent(obs_space, action_space, device="cpu")
    state = agent.get_state()

    # Act
    new_agent = AgentFactory.from_state(state)

    # Assert
    assert id(new_agent) != id(agent)
    assert new_agent == agent
    assert new_agent.model == PPOAgent.model
    assert new_agent.hparams == agent.hparams
    assert new_agent.buffer == agent.buffer


def test_agent_factory_ppo_agent_from_state_network_buffer_none():
    # Assign
    obs_space = DataSpace(dtype="float", shape=(10,))
    action_space = DataSpace(dtype="float", shape=(5,))
    agent = PPOAgent(obs_space, action_space, device="cpu")
    state = agent.get_state()
    state.network = None
    state.buffer = None

    # Act
    new_agent = AgentFactory.from_state(state)

    # Assert
    assert id(new_agent) != id(agent)
    assert new_agent.model == PPOAgent.model
    assert new_agent.hparams == agent.hparams


def test_agent_factory_ddpg_agent_from_state():
    # Assign
    obs_space = DataSpace(dtype="float", shape=(4,))
    action_space = DataSpace(dtype="float", shape=(4,))
    agent = DDPGAgent(obs_space, action_space, device="cpu")
    state = agent.get_state()

    # Act
    new_agent = AgentFactory.from_state(state)

    # Assert
    assert id(new_agent) != id(agent)
    assert new_agent.model == DDPGAgent.model
    assert new_agent == agent
    assert new_agent.hparams == agent.hparams
    assert new_agent.buffer == agent.buffer


def test_agent_factory_ddpg_agent_from_state_network_buffer_none():
    # Assign
    obs_space = DataSpace(dtype="float", shape=(4,))
    action_space = DataSpace(dtype="float", shape=(4,))
    agent = DDPGAgent(obs_space, action_space, device="cpu")
    state = agent.get_state()
    state.network = None
    state.buffer = None

    # Act
    new_agent = AgentFactory.from_state(state)

    # Assert
    assert id(new_agent) != id(agent)
    assert new_agent.model == DDPGAgent.model
    assert new_agent.hparams == agent.hparams


def test_agent_factory_sac_agent_from_state():
    # Assign
    obs_space = DataSpace(dtype="float", shape=(10,))
    action_space = DataSpace(dtype="float", shape=(5,))
    agent = SACAgent(obs_space, action_space, device="cpu")
    state = agent.get_state()

    # Act
    new_agent = AgentFactory.from_state(state)

    # Assert
    assert id(new_agent) != id(agent)
    assert new_agent == agent
    assert new_agent.model == SACAgent.model
    assert new_agent.hparams == agent.hparams
    assert new_agent.buffer == agent.buffer
