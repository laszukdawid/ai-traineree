import copy

from ai_traineree.agents.dqn import DQNAgent
from conftest import deterministic_interactions


def test_dqn_seed():
    # Assign
    agent_0 = DQNAgent(4, 4, device='cpu')  # Reference
    agent_1 = DQNAgent(4, 4, device='cpu')
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
    # All generated actions need to identical
    for idx, (a1, a2) in enumerate(zip(actions_1, actions_2)):
        assert a1 == a2, f"Action mismatch on position {idx}: {a1} != {a2}"
