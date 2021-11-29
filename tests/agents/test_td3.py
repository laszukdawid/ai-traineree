import copy

import pytest

from ai_traineree.agents.td3 import TD3Agent
from conftest import deterministic_interactions


def test_td3_seed(float_1d_space):
    # Assign
    agent_0 = TD3Agent(float_1d_space, float_1d_space, device="cpu")
    agent_1 = TD3Agent(float_1d_space, float_1d_space, device="cpu")
    agent_2 = copy.deepcopy(agent_1)

    # Act
    # Make sure agents have the same networks
    zip_agent_actors = zip(agent_1.actor.layers, agent_2.actor.layers)
    zip_agent_critics = zip(agent_1.critic.critic_1.layers, agent_2.critic.critic_1.layers)
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
