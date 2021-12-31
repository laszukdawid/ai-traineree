import copy

import pytest
import torch

from ai_traineree.agents.sac import SACAgent
from ai_traineree.types import DataSpace
from conftest import deterministic_interactions, feed_agent

float_space = DataSpace(dtype="float", shape=(5,), low=-2, high=2)
action_space = DataSpace(dtype="float", shape=(4,), low=-1, high=2)


def test_sac_seed(float_1d_space):
    # Assign
    agent_0 = SACAgent(float_1d_space, float_1d_space, device="cpu")  # Reference
    agent_1 = SACAgent(float_1d_space, float_1d_space, device="cpu")
    agent_2 = copy.deepcopy(agent_1)

    # Act
    # Make sure agents have the same networks
    zip_agent_actors = zip(agent_1.actor.layers, agent_2.actor.layers)
    zip_agent_critics = zip(agent_1.double_critic.critic_1.layers, agent_2.double_critic.critic_1.layers)
    assert all([sum(sum(l1.weight - l2.weight)) == 0 for l1, l2 in zip_agent_actors])
    assert all([sum(sum(l1.weight - l2.weight)) == 0 for l1, l2 in zip_agent_critics])

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


def test_sac_from_state():
    # Assign
    agent = SACAgent(float_space, action_space)
    agent_state = agent.get_state()

    # Act
    new_agent = SACAgent.from_state(agent_state)

    # Assert
    assert id(agent) != id(new_agent)
    # assert new_agent == agent
    assert isinstance(new_agent, SACAgent)
    assert new_agent.hparams == agent.hparams
    assert all([torch.all(x == y) for (x, y) in zip(agent.actor.parameters(), new_agent.actor.parameters())])
    assert all([torch.all(x == y) for (x, y) in zip(agent.policy.parameters(), new_agent.policy.parameters())])
    assert all(
        [torch.all(x == y) for (x, y) in zip(agent.double_critic.parameters(), new_agent.double_critic.parameters())]
    )
    assert all(
        [
            torch.all(x == y)
            for (x, y) in zip(agent.target_double_critic.parameters(), new_agent.target_double_critic.parameters())
        ]
    )
    assert new_agent.buffer == agent.buffer


def test_sac_from_state_network_state_none():
    # Assign
    agent = SACAgent(float_space, action_space)
    agent_state = agent.get_state()
    agent_state.network = None

    # Act
    new_agent = SACAgent.from_state(agent_state)

    # Assert
    assert id(agent) != id(new_agent)
    # assert new_agent == agent
    assert isinstance(new_agent, SACAgent)
    assert new_agent.hparams == agent.hparams
    assert new_agent.buffer == agent.buffer


def test_sac_from_state_buffer_state_none():
    # Assign
    agent = SACAgent(float_space, float_space)
    agent_state = agent.get_state()
    agent_state.buffer = None

    # Act
    new_agent = SACAgent.from_state(agent_state)

    # Assert
    assert id(agent) != id(new_agent)
    # assert new_agent == agent
    assert isinstance(new_agent, SACAgent)
    assert new_agent.hparams == agent.hparams
    assert all([torch.all(x == y) for (x, y) in zip(agent.actor.parameters(), new_agent.actor.parameters())])
    assert all([torch.all(x == y) for (x, y) in zip(agent.policy.parameters(), new_agent.policy.parameters())])
    assert all(
        [torch.all(x == y) for (x, y) in zip(agent.double_critic.parameters(), new_agent.double_critic.parameters())]
    )
    assert all(
        [
            torch.all(x == y)
            for (x, y) in zip(agent.target_double_critic.parameters(), new_agent.target_double_critic.parameters())
        ]
    )


def test_sac_from_state_one_updated():
    # Assign
    agent = SACAgent(float_space, float_space)
    feed_agent(agent, 2 * agent.batch_size)  # Feed 1
    agent_state = agent.get_state()
    feed_agent(agent, 100)  # Feed 2 - to make different

    # Act
    new_agent = SACAgent.from_state(agent_state)

    # Assert
    assert id(agent) != id(new_agent)
    # assert new_agent == agent
    assert isinstance(new_agent, SACAgent)
    assert new_agent.hparams == agent.hparams
    assert all([torch.any(x != y) for (x, y) in zip(agent.actor.parameters(), new_agent.actor.parameters())])
    assert all([torch.all(x != y) for (x, y) in zip(agent.policy.parameters(), new_agent.policy.parameters())])
    assert all(
        [torch.any(x != y) for (x, y) in zip(agent.double_critic.parameters(), new_agent.double_critic.parameters())]
    )
    assert all(
        [
            torch.any(x != y)
            for (x, y) in zip(agent.target_double_critic.parameters(), new_agent.target_double_critic.parameters())
        ]
    )
    assert new_agent.buffer != agent.buffer
