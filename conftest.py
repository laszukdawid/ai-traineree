import mock
import numpy as np
import pytest


class MockContinuousSpace:
    def __init__(self, *args):
        self.shape = args


class MockDiscreteSpace:
    def __init__(self, n):
        self.n = n

    def __call__(self):
        return self.n


@pytest.fixture
def fix_env_discrete():
    import random
    rnd_state = lambda: random.choices(range(10), k=5)  # noqa
    mock_env = mock.Mock()
    mock_env.reset.return_value = rnd_state()
    mock_env.step.return_value = (rnd_state(), 0, False, "")
    mock_env.observation_space = np.array((4, 2))
    mock_env.action_space = MockDiscreteSpace(2)
    return mock_env


@pytest.fixture
def fix_env():
    import random
    rnd_state = lambda: random.choices(range(10), k=5)  # noqa
    mock_env = mock.Mock()
    mock_env.reset.return_value = rnd_state()
    mock_env.step.return_value = (rnd_state(), 0, False, "")
    mock_env.observation_space = np.array((4, 2))
    mock_env.action_space = MockContinuousSpace(2, 4)
    return mock_env
