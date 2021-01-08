import copy
import pytest

from ai_traineree.networks.heads import RainbowNet
from ai_traineree.agents.rainbow import RainbowAgent
from conftest import deterministic_interactions


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
    assert agent.in_features == input_shape[0]
    assert agent.out_features == output_shape[0]


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
