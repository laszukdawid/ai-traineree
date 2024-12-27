import random
from typing import List

import mock
import pytest

from ai_traineree.agents.ppo import PPOAgent
from ai_traineree.runners.multi_sync_env_runner import MultiSyncEnvRunner
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType

# Disable the whole unit test
# TODO: There is something wrong with processes; multi processes are not working with 3.12
pytestmark = pytest.mark.skip

# NOTE: Some of these tests use `test_task` and `test_agent` which are real instances.
#       This is partially to make sure that the tricky part is covered, and not hid
#       by aggressive mocking. The other part, however, is the burden of keeping env mocks.
#       This results in unnecessary performance hit. A lightweight env would be nice.

test_task = GymTask("LunarLanderContinuous-v3")
test_agent = PPOAgent(test_task.obs_space, test_task.action_space)


@mock.patch("ai_traineree.runners.multi_sync_env_runner.AgentBase")
@mock.patch("ai_traineree.runners.multi_sync_env_runner.TaskType")
def test_multi_sync_env_runner_init_str_check(mock_task, mock_agent):
    # Assign & Act
    mock_agent.model = "Agent"
    mock_task.name = "Task"
    multi_sync_env_runner = MultiSyncEnvRunner([mock_task], mock_agent)

    # Assert
    assert str(multi_sync_env_runner) == "MultiSyncEnvRunner<['Task'], Agent>"


@mock.patch("ai_traineree.runners.multi_sync_env_runner.AgentBase")
@mock.patch("ai_traineree.runners.multi_sync_env_runner.TaskType")
def test_multi_sync_env_runner_reset(mock_task, mock_agent):
    # Assign
    multi_sync_env_runner = MultiSyncEnvRunner([mock_task], mock_agent, window_len=10)
    multi_sync_env_runner.episode = 10
    multi_sync_env_runner.all_iterations.extend(map(lambda _: random.randint(1, 100), range(10)))
    multi_sync_env_runner.all_scores.extend(map(lambda _: random.random(), range(10)))
    multi_sync_env_runner.scores_window.extend(map(lambda _: random.random(), range(10)))

    # Act
    multi_sync_env_runner.reset()

    # Assert
    assert multi_sync_env_runner.episode == 0
    assert len(multi_sync_env_runner.all_iterations) == 0
    assert len(multi_sync_env_runner.all_scores) == 0
    assert len(multi_sync_env_runner.scores_window) == 0


def test_multi_sync_env_runner_run_single_step_single_task():
    # Assign
    multi_sync_env_runner = MultiSyncEnvRunner([test_task], test_agent)

    # Act
    scores = multi_sync_env_runner.run(max_episodes=1, max_iterations=1, force_new=True)

    # Assert
    assert len(scores) == 1  # No chance that it'll terminate episode in 1 iteration


def test_multi_sync_env_runner_run_single_step_multiple_task():
    # Assign
    tasks: List[TaskType] = [test_task, test_task]
    agent = PPOAgent(test_task.obs_space, test_task.action_space, num_workers=len(tasks))
    multi_sync_env_runner = MultiSyncEnvRunner(tasks, agent)

    # Act
    scores = multi_sync_env_runner.run(max_episodes=1, max_iterations=1, force_new=True)

    # Assert
    assert len(scores) == 2  # After 1 iteration both "finished" at the same time


def test_multi_sync_env_runner_run_multiple_step_multiple_task():
    # Assign
    tasks: List[TaskType] = [test_task, test_task]
    agent = PPOAgent(test_task.obs_space, test_task.action_space, num_workers=len(tasks))
    multi_sync_env_runner = MultiSyncEnvRunner(tasks, agent)

    # Act
    scores = multi_sync_env_runner.run(max_episodes=3, max_iterations=100, force_new=True)

    # Assert
    assert len(scores) in (3, 4)  # On rare occasions two tasks can complete twice at the same time.


@mock.patch("ai_traineree.runners.multi_sync_env_runner.AgentBase")
@mock.patch("ai_traineree.runners.multi_sync_env_runner.TaskType")
def test_multi_sync_env_runner_info_no_data_logger(mock_task, mock_agent):
    # Assign
    mock_tasks: List[TaskType] = [mock_task, mock_task]
    env_runner = MultiSyncEnvRunner(mock_tasks, mock_agent)
    env_runner.logger = mock.MagicMock()
    info_data = dict(episodes=[2], iterations=[10], scores=[1], mean_scores=[2], epsilons=[1])

    # Act
    env_runner.info(**info_data)

    # Assert
    env_runner.logger.info.assert_called_once()


@mock.patch("ai_traineree.runners.multi_sync_env_runner.AgentBase")
@mock.patch("ai_traineree.runners.multi_sync_env_runner.TaskType")
def test_multi_sync_env_runner_info_with_data_logger(mock_task, mock_agent):
    # Assign
    data_logger = mock.MagicMock()
    env_runner = MultiSyncEnvRunner(mock_task, mock_agent, data_logger=data_logger)
    env_runner.logger = mock.MagicMock()
    info_data = dict(episodes=[2], iterations=[10], scores=[1], mean_scores=[2], epsilons=[1])

    # Act
    env_runner.info(**info_data)

    # Assert
    env_runner.logger.info.assert_called_once()
    assert data_logger.log_value.call_count == 4
    mock_agent.log_metrics.assert_called_once_with(data_logger, mock.ANY, full_log=False)


@mock.patch("ai_traineree.runners.multi_sync_env_runner.AgentBase")
@mock.patch("ai_traineree.runners.multi_sync_env_runner.TaskType")
@mock.patch("ai_traineree.runners.multi_sync_env_runner.DataLogger")
def test_multi_sync_env_runner_log_episode_metrics(mock_data_logger, mock_task, mock_agent):
    # Assign
    episodes = [1, 2]
    epsilons = [0.2, 0.1]
    mean_scores = [0.5, 1]
    scores = [1.5, 5]
    iterations = [10, 10]
    episode_data = dict(
        episodes=episodes, epsilons=epsilons, mean_scores=mean_scores, iterations=iterations, scores=scores
    )
    env_runner = MultiSyncEnvRunner(mock_task, mock_agent, data_logger=mock_data_logger)

    # Act
    env_runner.log_episode_metrics(**episode_data)

    # Assert
    for idx, episode in enumerate(episodes):
        mock_data_logger.log_value.assert_any_call("episode/epsilon", epsilons[idx], episode)
        mock_data_logger.log_value.assert_any_call("episode/avg_score", mean_scores[idx], episode)
        mock_data_logger.log_value.assert_any_call("episode/score", scores[idx], episode)
        mock_data_logger.log_value.assert_any_call("episode/iterations", iterations[idx], episode)


@mock.patch("ai_traineree.runners.multi_sync_env_runner.AgentBase")
@mock.patch("ai_traineree.runners.multi_sync_env_runner.TaskType")
@mock.patch("ai_traineree.runners.multi_sync_env_runner.DataLogger")
def test_multi_sync_env_runner_log_episode_metrics_values_missing(mock_data_logger, mock_task, mock_agent):
    # Assign
    episodes = [1, 2]
    episode_data = dict(episodes=episodes)
    env_runner = MultiSyncEnvRunner(mock_task, mock_agent, data_logger=mock_data_logger)

    # Act
    env_runner.log_episode_metrics(**episode_data)

    # Assert
    mock_data_logger.log_value.assert_not_called()


@mock.patch("ai_traineree.runners.multi_sync_env_runner.AgentBase")
@mock.patch("ai_traineree.runners.multi_sync_env_runner.TaskType")
def test_multi_sync_env_runner_log_data_interaction_no_data_logger(mock_task, mock_agent):
    # Assign
    env_runner = MultiSyncEnvRunner(mock_task, mock_agent)

    # Act
    env_runner.log_data_interaction()

    # Assert
    mock_agent.log_metrics.assert_not_called()


@mock.patch("ai_traineree.runners.multi_sync_env_runner.DataLogger")
def test_multi_sync_env_runner_log_data_interaction_iterations(mock_data_logger):
    # Assign
    test_agent.log_metrics = mock.MagicMock()
    env_runner = MultiSyncEnvRunner([test_task], test_agent, data_logger=mock_data_logger)

    # Act
    env_runner.run(max_episodes=1, max_iterations=10, log_episode_freq=2, force_new=True)
    env_runner.log_data_interaction()

    # Assert
    test_agent.log_metrics.assert_called_once_with(mock_data_logger, 10, full_log=False)
    assert mock_data_logger.log_values_dict.call_count == 0
    assert mock_data_logger.log_value.call_count == 0


@mock.patch("ai_traineree.runners.multi_sync_env_runner.DataLogger")
def test_multi_sync_env_runner_log_data_interaction_log_after_episode(mock_data_logger):
    # Assign
    test_agent.log_metrics = mock.MagicMock()
    env_runner = MultiSyncEnvRunner([test_task], test_agent, data_logger=mock_data_logger)

    # Act
    env_runner.run(max_episodes=1, max_iterations=10, force_new=True)

    # Assert
    test_agent.log_metrics.assert_called_once_with(mock_data_logger, 10, full_log=False)
    assert mock_data_logger.log_values_dict.call_count == 0
    assert mock_data_logger.log_value.call_count == 4


@mock.patch("ai_traineree.runners.multi_sync_env_runner.Path")
@mock.patch("ai_traineree.runners.multi_sync_env_runner.json")
def test_multi_sync_env_runner_save_state(mock_json, mock_path):
    # Assign
    test_agent.save_state = mock.MagicMock()
    env_runner = MultiSyncEnvRunner([test_task], test_agent)

    # Act
    env_runner.run(max_episodes=10, max_iterations=10, force_new=True)
    with mock.patch("builtins.open"):
        env_runner.save_state("saved_state.state")

    # Assert
    test_agent.save_state.assert_called_once()
    state = mock_json.dump.call_args[0][0]
    assert state["episode"] == 10
    assert state["tot_iterations"] == 10 * 10


@mock.patch("ai_traineree.runners.multi_sync_env_runner.AgentBase")
@mock.patch("ai_traineree.runners.multi_sync_env_runner.TaskType")
def test_multi_sync_env_runner_load_state_no_file(mock_task, mock_agent):
    # Assign
    env_runner = MultiSyncEnvRunner(mock_task, mock_agent, max_iterations=10)
    env_runner.logger = mock.MagicMock()

    # Act
    env_runner.load_state(file_prefix="saved_state")

    # Assert
    env_runner.logger.warning.assert_called_once_with("Couldn't load state. Forcing restart.")
    mock_agent.load_state.assert_not_called()


@mock.patch("ai_traineree.runners.multi_sync_env_runner.os")
@mock.patch("ai_traineree.runners.multi_sync_env_runner.AgentBase")
@mock.patch("ai_traineree.runners.multi_sync_env_runner.TaskType")
def test_multi_sync_env_runner_load_state(mock_task, mock_agent, mock_os):
    # Assign
    env_runner = MultiSyncEnvRunner(mock_task, mock_agent, max_iterations=10)
    mock_os.listdir.return_value = ["saved_state_e10.json", "saved_state_e999.json", "other.file"]
    mocked_state = '{"episode": 10, "epsilon": 0.2, "score": 0.3, "average_score": -0.1}'

    # Act
    with mock.patch("builtins.open", mock.mock_open(read_data=mocked_state)) as mock_file:
        env_runner.load_state(file_prefix="saved_state")
        mock_file.assert_called_once_with(f"{env_runner.state_dir}/saved_state_e999.json", "r")

    # Assert
    mock_agent.load_state.assert_called_once()
    assert env_runner.episode == 10
    assert env_runner.epsilon == 0.2
    assert len(env_runner.all_scores) == 1
    assert env_runner.all_scores[0] == 0.3
