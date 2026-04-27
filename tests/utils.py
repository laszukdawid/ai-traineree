import copy
import random
from typing import Any, Sequence

import numpy as np

from aitraineree.agents import AgentBase
from aitraineree.types.agent import AgentType
from aitraineree.types.experience import Experience


def generate_sample_SARS(iterations, obs_size: int = 4, action_size: int = 2, dict_type=False):
    def state_fn():
        return np.random.random(obs_size)

    def action_fn():
        return np.random.random(action_size)

    def reward_fn():
        return float(np.random.random() - 0.5)

    def done_fn():
        return np.random.random() > 0.5

    state = state_fn()

    for _ in range(iterations):
        next_state = state_fn()
        if dict_type:
            yield dict(
                state=list(state),
                action=list(action_fn()),
                reward=[reward_fn()],
                next_state=list(next_state),
                done=[bool(done_fn())],
            )
        else:
            yield (list(state), list(action_fn()), reward_fn(), list(next_state), bool(done_fn()))
        state = next_state


def deterministic_interactions(agent: AgentType, num_iters=50) -> list[Any]:
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


def fake_step(step_shape: Sequence[int]) -> tuple[list[Any], float, bool, bool]:
    state = np.random.random(step_shape).tolist()
    reward = random.random()
    terminal = random.random() > 0.8
    truncated = False
    return state, reward, terminal, truncated


def feed_agent(agent: AgentBase, num_samples: int, as_list=False):
    action_space = agent.action_space
    s, _, _, _ = fake_step(agent.obs_space.shape)

    for _ in range(num_samples):
        sn, r, ter, trunc = fake_step(agent.obs_space.shape)
        d = ter or trunc
        if action_space.dtype == "int":
            a = int(np.random.randint(0, action_space.shape)[0])  # Only one action allowed
        else:
            a = np.random.random(action_space.shape)

        if as_list:
            experience = Experience(obs=s, action=[a], reward=[r], next_obs=sn, done=[d])
            agent.step(experience)
        else:
            experience = Experience(obs=s, action=a, reward=r, next_obs=sn, done=d)
            agent.step(experience)
        s = sn
    return agent
