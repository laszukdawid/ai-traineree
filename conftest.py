import copy
import random
from math import exp
from typing import Any, List, Sequence, Tuple

import mock
import numpy as np
import pytest

from ai_traineree.agents import AgentBase
from ai_traineree.types.agent import AgentType
from ai_traineree.types.dataspace import DataSpace
from ai_traineree.types.experience import Experience


class MockContinuousSpace:
    def __init__(self, *args):
        self.shape = args


class MockDiscreteSpace:
    def __init__(self, n):
        self.n = n

    def __call__(self):
        return self.n


def rnd_state():
    return random.choices(range(10), k=5)


@pytest.fixture
def fix_env_discrete():
    mock_env = mock.Mock()
    mock_env.reset.return_value = rnd_state()
    mock_env.step.return_value = (rnd_state(), 0, False, "")
    mock_env.observation_space = np.array((4, 2))
    mock_env.action_space = MockDiscreteSpace(2)
    return mock_env


@pytest.fixture
def fix_env():
    mock_env = mock.Mock()
    mock_env.reset.return_value = rnd_state()
    mock_env.step.return_value = (rnd_state(), 0, False, "")
    mock_env.observation_space = np.array((4, 2))
    mock_env.action_space = MockContinuousSpace(2, 4)
    return mock_env


def deterministic_interactions(agent: AgentType, num_iters=50):
    obs_size = agent.obs_space.shape[0]
    obs = np.zeros(agent.obs_space.shape).tolist()
    next_obs = copy.copy(obs)
    actions = []
    for i in range(num_iters):
        experience = Experience(obs=obs)
        experience = agent.act(experience)
        actions.append(experience.action)

        next_obs[i % obs_size] = (next_obs[i % obs_size] + 1) % 2
        reward = (i % 4 - 2) / 2.0
        done = (i + 1) % 100 == 0
        experience.update(done=done, reward=reward, next_obs=next_obs)

        agent.step(experience)
        obs = copy.copy(next_obs)
    return actions


def fake_step(step_shape: Sequence[int]) -> Tuple[List[Any], float, bool]:
    state = np.random.random(step_shape).tolist()
    reward = random.random()
    terminal = random.random() > 0.8
    return state, reward, terminal


def feed_agent(agent: AgentBase, num_samples: int, as_list=False):
    action_space = agent.action_space

    for _ in range(num_samples):
        s, r, d = fake_step(agent.obs_space.shape)
        if action_space.dtype == "int":
            # a = random.randint(0, agent.action_size-1)
            a = int(np.random.randint(0, action_space.shape)[0])  # Only one action allowed
        else:
            # a = np.random.random(action_size).tolist()
            a = np.random.random(action_space.shape)

        if as_list:
            experience = Experience(obs=s, action=[s], reward=[r], next_obs=s, done=[d])
            agent.step(experience)
        else:
            experience = Experience(obs=s, action=a, reward=r, next_obs=s, done=d)
            agent.step(experience)
    return agent


@pytest.fixture
def float_1d_space():
    return DataSpace(dtype="float", shape=(5,), low=-2, high=2)


@pytest.fixture
def int_1d_space():
    return DataSpace(dtype="int", shape=(1,), low=0, high=4)
