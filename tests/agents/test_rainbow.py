import pytest

from ai_traineree.networks.heads import RainbowNet
from ai_traineree.agents.rainbow import RainbowAgent


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

