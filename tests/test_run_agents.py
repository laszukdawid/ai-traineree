"""
This 'Test suite' simply makes sure that all/most agents run.
Runs might not be perfect, might not be meaningful but at least
the code isn't broken.

**DO NOT TREAT THIS AS A UNIT TEST**
"""

import pytest

from aitraineree.agents.d3pg import D3PGAgent
from aitraineree.agents.ddpg import DDPGAgent
from aitraineree.agents.dqn import DQNAgent
from aitraineree.agents.ppo import PPOAgent
from aitraineree.agents.rainbow import RainbowAgent
from aitraineree.agents.sac import SACAgent
from aitraineree.agents.td3 import TD3Agent
from aitraineree.runners.env_runner import EnvRunner
from aitraineree.tasks import GymTask

DEVICE = "cpu"


# Discrete agents
def test_runs_dqn():
    # Assign
    task = GymTask("CartPole-v1")
    agent = DQNAgent(task.obs_space, task.action_space, device=DEVICE)
    env_runner = EnvRunner(task, agent, max_iterations=50)

    # Act
    env_runner.run(reward_goal=10, max_episodes=10, force_new=True)


def test_runs_rainbow():
    # Assign
    task = GymTask("CartPole-v1")
    agent = RainbowAgent(task.obs_space, task.action_space, device=DEVICE)
    env_runner = EnvRunner(task, agent, max_iterations=50)

    # Act
    env_runner.run(reward_goal=10, max_episodes=10, force_new=True)


def test_runs_ppo():
    # Assign
    task = GymTask("Pendulum-v1")
    agent = PPOAgent(task.obs_space, task.action_space, device=DEVICE)
    env_runner = EnvRunner(task, agent, max_iterations=50)

    # Act
    env_runner.run(reward_goal=10, max_episodes=10, force_new=True)


def test_runs_ddpg():
    # Assign
    task = GymTask("Pendulum-v1")
    agent = DDPGAgent(task.obs_space, task.action_space, device=DEVICE)
    env_runner = EnvRunner(task, agent, max_iterations=50)

    # Act
    env_runner.run(reward_goal=10, max_episodes=10, force_new=True)


def test_runs_td3():
    # Assign
    task = GymTask("Pendulum-v1")
    agent = TD3Agent(task.obs_space, task.action_space, device=DEVICE)
    env_runner = EnvRunner(task, agent, max_iterations=50)

    # Act
    env_runner.run(reward_goal=10, max_episodes=10, force_new=True)


def test_runs_sac():
    # Assign
    task = GymTask("BipedalWalker-v3")
    agent = SACAgent(task.obs_space, task.action_space, device=DEVICE)
    env_runner = EnvRunner(task, agent, max_iterations=50)

    # Act
    env_runner.run(reward_goal=10, max_episodes=10, force_new=True)


@pytest.mark.skip("Currently broken!")
def test_runs_d3pg():
    # Assign
    task = GymTask("BipedalWalker-v3")
    agent = D3PGAgent(task.obs_space, task.action_space, device=DEVICE)
    env_runner = EnvRunner(task, agent, max_iterations=50)

    # Act
    env_runner.run(reward_goal=10, max_episodes=10, force_new=True)
