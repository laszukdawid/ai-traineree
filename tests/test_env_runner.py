import random

from ai_traineree.agents.ppo import PPOAgent
from ai_traineree.env_runner import MultiSyncEnvRunner
from ai_traineree.tasks import GymTask

"""
Acronyms:
maer - MultiSyncEnvRunner
"""

test_task = GymTask('LunarLanderContinuous-v2')
test_agent = PPOAgent(test_task.state_size, test_task.action_size)


def test_maer_init_str_check():
    # Assign & Act
    maer = MultiSyncEnvRunner([test_task], test_agent)

    # Assert
    assert str(maer) == f"MultiSyncEnvRunner<['{test_task.name}'], {test_agent.name}>"


def test_maer_reset():
    # Assign
    maer = MultiSyncEnvRunner([test_task], test_agent, window_len=10)
    maer.episode = 10
    maer.all_iterations.extend(map(lambda _: random.randint(1, 100), range(10)))
    maer.all_scores.extend(map(lambda _: random.random(), range(10)))
    maer.scores_window.extend(map(lambda _: random.random(), range(10)))

    # Act
    maer.reset()

    # Assert
    assert maer.episode == 0
    assert len(maer.all_iterations) == 0
    assert len(maer.all_scores) == 0
    assert len(maer.scores_window) == 0


def test_maer_run_single_step_single_task():
    # Assign
    maer = MultiSyncEnvRunner([test_task], test_agent)

    # Act
    scores = maer.run(max_episodes=1, max_iterations=1, force_new=True)

    # Assert
    assert len(scores) == 1  # No chance that it'll terminate episode in 1 iteration


def test_maer_run_single_step_multiple_task():
    # Assign
    tasks = [test_task, test_task]
    agent = PPOAgent(test_task.state_size, test_task.action_size, executor_num=len(tasks))
    maer = MultiSyncEnvRunner(tasks, agent)

    # Act
    scores = maer.run(max_episodes=1, max_iterations=1, force_new=True)

    # Assert
    assert len(scores) == 2  # After 1 iteration both "finished" at the same time


def test_maer_run_multiple_step_multiple_task():
    # Assign
    tasks = [test_task, test_task]
    agent = PPOAgent(test_task.state_size, test_task.action_size, executor_num=len(tasks))
    maer = MultiSyncEnvRunner(tasks, agent)

    # Act
    scores = maer.run(max_episodes=3, max_iterations=100, force_new=True)

    # Assert
    assert len(scores) in (3, 4)  # On rare occasions two tasks can complete twice at the same time.
