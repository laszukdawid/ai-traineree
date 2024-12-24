import random

import mock
import pytest

from ai_traineree.multi_agents.maddpg import MADDPGAgent
from ai_traineree.runners.multiagent_env_runner import MultiAgentCycleEnvRunner
from ai_traineree.tasks import PettingZooTask

pettingzoo = pytest.importorskip("pettingzoo")
multiwalker_v7 = pettingzoo.multiwalker_v7

# NOTE: Some of these tests use `test_task` and `test_agent` which are real instances.
#       This is partially to make sure that the tricky part is covered, and not hid
#       by aggressive mocking. The other part, however, is the burden of keeping env mocks.
#       This results in unnecessary performance hit. A lightweight env would be nice.

env = multiwalker_v7.env()
test_task = PettingZooTask(env)
test_task.reset()
obs_space = list(test_task.observation_spaces.values())[0]
action_space = list(test_task.action_spaces.values())[0]
test_agent = MADDPGAgent(obs_space, action_space, num_agents=env.num_agents, agent_names=env.agents)


def test_multiagent_cycle_env_runner_str():
    # Assign
    env_runner = MultiAgentCycleEnvRunner(test_task, test_agent)

    # Act & Assert
    assert str(env_runner) == f"MultiAgentCycleEnvRunner<{test_task.name}, {test_agent.model}>"


@mock.patch("ai_traineree.runners.multiagent_env_runner.MultiAgentType")
@mock.patch("ai_traineree.runners.multiagent_env_runner.MultiAgentTaskType")
def test_multiagent_cycle_env_runner_seed(mock_task, mock_agent):
    # Assign
    env_runner = MultiAgentCycleEnvRunner(mock_task, mock_agent)
    seed = 32167

    # Act
    env_runner.seed(seed)

    # Assert
    mock_agent.seed.assert_called_once_with(seed)
    mock_task.seed.assert_called_once_with(seed)


@mock.patch("ai_traineree.runners.multiagent_env_runner.MultiAgentType")
@mock.patch("ai_traineree.runners.multiagent_env_runner.MultiAgentTaskType")
def test_multiagent_cycle_env_runner_reset(mock_task, mock_agent):
    # Assign
    multi_sync_env_runner = MultiAgentCycleEnvRunner(mock_task, mock_agent, window_len=10)
    multi_sync_env_runner.episode = 10
    multi_sync_env_runner.all_iterations.extend(map(lambda _: random.randint(1, 100), range(10)))
    multi_sync_env_runner.all_scores.extend(map(lambda _: {str(i): random.random() for i in range(2)}, range(10)))
    multi_sync_env_runner.scores_window.extend(map(lambda _: random.random(), range(10)))

    # Act
    multi_sync_env_runner.reset()

    # Assert
    assert multi_sync_env_runner.episode == 0
    assert len(multi_sync_env_runner.all_iterations) == 0
    assert len(multi_sync_env_runner.all_scores) == 0
    assert len(multi_sync_env_runner.scores_window) == 0


def test_multiagent_cycle_env_runner_interact_episode():
    # Assign
    test_task.render = mock.MagicMock()
    multi_sync_env_runner = MultiAgentCycleEnvRunner(test_task, test_agent, max_iterations=10)

    # Act
    output = multi_sync_env_runner.interact_episode()

    # Assert
    assert len(output) == 2  # (rewards, iterations)
    assert isinstance(output[0], dict)
    assert len(output[0]) == test_agent.num_agents
    assert output[1] > 1

    assert len(multi_sync_env_runner._images) == 0
    assert len(multi_sync_env_runner._actions) == 0
    assert len(multi_sync_env_runner._rewards) == 0
    assert len(multi_sync_env_runner._dones) == 0

    test_task.render.assert_not_called()


def test_multiagent_cycle_env_runner_interact_episode_override_max_iteractions():
    # Assign
    test_task.render = mock.MagicMock()
    multi_sync_env_runner = MultiAgentCycleEnvRunner(test_task, test_agent, max_iterations=10)

    # Act
    _, interactions = multi_sync_env_runner.interact_episode(max_iterations=20)

    # Assert
    assert interactions == 20


def test_multiagent_cycle_env_runner_interact_episode_render_gif():
    # Assign
    test_task.render = mock.MagicMock(return_value=[[0, 0, 1], [0, 1, 0], [1, 1, 0]])
    multi_sync_env_runner = MultiAgentCycleEnvRunner(test_task, test_agent, max_iterations=10)

    # Act
    multi_sync_env_runner.interact_episode(render_gif=True)

    # Assert
    assert len(multi_sync_env_runner._images) == 10
    assert test_task.render.call_count == 10


def test_multiagent_cycle_env_runner_interact_episode_debug_log():
    # Assign
    multi_sync_env_runner = MultiAgentCycleEnvRunner(test_task, test_agent, max_iterations=10, debug_log=True)

    # Act
    multi_sync_env_runner.interact_episode()

    # Assert
    assert all([len(actions) == 10 for actions in multi_sync_env_runner._actions.values()])
    assert all([len(dones) == 10 for dones in multi_sync_env_runner._dones.values()])
    assert all([len(rewards) == 10 for rewards in multi_sync_env_runner._rewards.values()])


def test_multiagent_cycle_env_runner_interact_episode_log_interaction_without_data_logger():
    # Assign
    test_agent.log_metrics = mock.MagicMock()
    multi_sync_env_runner = MultiAgentCycleEnvRunner(test_task, test_agent, max_iterations=10)
    multi_sync_env_runner.log_data_interaction = mock.MagicMock()

    # Act
    multi_sync_env_runner.interact_episode(log_interaction_freq=1)

    # Assert
    assert multi_sync_env_runner.log_data_interaction.call_count == 10
    assert test_agent.log_metrics.call_count == 0


@mock.patch("ai_traineree.runners.multiagent_env_runner.DataLogger")
def test_multiagent_cycle_env_runner_interact_episode_log_interaction(mock_data_logger):
    # Assign
    test_agent.log_metrics = mock.MagicMock()
    multi_sync_env_runner = MultiAgentCycleEnvRunner(
        test_task, test_agent, data_logger=mock_data_logger, max_iterations=10
    )

    # Act
    multi_sync_env_runner.interact_episode(log_interaction_freq=1)

    # Assert
    assert test_agent.log_metrics.call_count == 10
    test_agent.log_metrics.assert_called_with(mock_data_logger, 10, full_log=False)  # Last
    mock_data_logger.log_value.assert_not_called()
    mock_data_logger.log_value_dict.assert_not_called()


def test_multiagent_cycle_env_runner_run():
    # Assign
    return_rewards = {name: 1 for name in test_agent.agents}
    multi_sync_env_runner = MultiAgentCycleEnvRunner(test_task, test_agent)
    multi_sync_env_runner.interact_episode = mock.MagicMock(return_value=(return_rewards, 10))

    # Act
    out = multi_sync_env_runner.run(max_episodes=5)

    # Assert
    assert multi_sync_env_runner.interact_episode.call_count == 5
    assert len(out) == 5
    assert len(out[0]) == test_agent.num_agents


@mock.patch("ai_traineree.runners.multiagent_env_runner.MultiAgentType")
@mock.patch("ai_traineree.runners.multiagent_env_runner.MultiAgentTaskType")
@mock.patch("ai_traineree.runners.multiagent_env_runner.DataLogger")
def test_multiagent_cycle_env_runner_log_episode_metrics(mock_data_logger, mock_task, mock_agent):
    # Assign
    episodes = [1, 2]
    epsilons = [0.2, 0.1]
    mean_scores = [0.5, 1]
    scores = [[1.5, 5], [2.0, 0.1]]
    iterations = [10, 10]
    episode_data = dict(
        episodes=episodes,
        epsilons=epsilons,
        mean_scores=mean_scores,
        iterations=iterations,
        scores=scores,
    )
    env_runner = MultiAgentCycleEnvRunner(mock_task, mock_agent, data_logger=mock_data_logger)

    # Act
    env_runner.log_episode_metrics(**episode_data)

    # Assert
    for idx, episode in enumerate(episodes):
        mock_data_logger.log_value.assert_any_call("episode/epsilon", epsilons[idx], episode)
        mock_data_logger.log_value.assert_any_call("episode/avg_score", mean_scores[idx], episode)
        mock_data_logger.log_value.assert_any_call("episode/iterations", iterations[idx], episode)
        mock_data_logger.log_values_dict.assert_any_call("episode/score", scores[idx], episode)


@mock.patch("ai_traineree.runners.multiagent_env_runner.MultiAgentType")
@mock.patch("ai_traineree.runners.multiagent_env_runner.MultiAgentTaskType")
@mock.patch("ai_traineree.runners.multiagent_env_runner.DataLogger")
def test_multiagent_cycle_env_runner_log_data_interaction(mock_data_logger, mock_task, mock_agent):
    # Assign
    env_runner = MultiAgentCycleEnvRunner(mock_task, mock_agent, data_logger=mock_data_logger)

    # Act
    env_runner.log_data_interaction()

    # Assert
    mock_agent.log_metrics.assert_called_once_with(mock_data_logger, 0, full_log=False)


@mock.patch("ai_traineree.runners.multiagent_env_runner.MultiAgentType")
@mock.patch("ai_traineree.runners.multiagent_env_runner.MultiAgentTaskType")
def test_multiagent_cycle_env_runner_log_data_interaction_no_data_logger(mock_task, mock_agent):
    # Assign
    env_runner = MultiAgentCycleEnvRunner(mock_task, mock_agent)

    # Act
    env_runner.log_data_interaction()

    # Assert
    mock_agent.log_metrics.assert_not_called()


@mock.patch("ai_traineree.runners.multiagent_env_runner.DataLogger")
def test_multiagent_cycle_env_runner_log_data_interaction_debug_log(mock_data_logger):
    # Assign
    test_agent.log_metrics = mock.MagicMock()
    env_runner = MultiAgentCycleEnvRunner(test_task, test_agent, data_logger=mock_data_logger, debug_log=True)

    # Act
    env_runner.interact_episode(eps=0.1, max_iterations=10, log_interaction_freq=None)
    env_runner.log_data_interaction()

    # Assert
    test_agent.log_metrics.assert_called_once_with(mock_data_logger, 10, full_log=False)
    assert mock_data_logger.log_values_dict.call_count == 90  # 3 agents x (A + R + D) x 10 interactions
    assert mock_data_logger.log_value.call_count == 0  # 10x iter per rewards and dones


@mock.patch("ai_traineree.runners.multiagent_env_runner.Path")
@mock.patch("ai_traineree.runners.multiagent_env_runner.json")
@mock.patch("ai_traineree.runners.multiagent_env_runner.MultiAgentType")
@mock.patch("ai_traineree.runners.multiagent_env_runner.MultiAgentTaskType")
def test_multiagent_cycle_env_runner_save_state(mock_task, mock_agent, mock_json, mock_path):
    # Assign
    mock_task.step.return_value = ([1, 0.1], -1, False, {})
    mock_agent.act.return_value = 1
    env_runner = MultiAgentCycleEnvRunner(mock_task, mock_agent, max_iterations=10)

    # Act
    env_runner.run(max_episodes=10)
    with mock.patch("builtins.open"):
        env_runner.save_state("saved_state.state")

    # Assert
    mock_agent.save_state.assert_called_once()
    state = mock_json.dump.call_args[0][0]
    assert state["episode"] == 10
    assert state["tot_iterations"] == 10 * 10


@mock.patch("ai_traineree.runners.multiagent_env_runner.MultiAgentType")
@mock.patch("ai_traineree.runners.multiagent_env_runner.MultiAgentTaskType")
def test_multiagent_cycle_env_runner_load_state_no_file(mock_task, mock_agent):
    # Assign
    env_runner = MultiAgentCycleEnvRunner(mock_task, mock_agent, max_iterations=10)
    env_runner.logger = mock.MagicMock()

    # Act
    env_runner.load_state(file_prefix="saved_state")

    # Assert
    env_runner.logger.warning.assert_called_once_with("Couldn't load state. Forcing restart.")
    mock_agent.load_state.assert_not_called()


@mock.patch("ai_traineree.runners.multiagent_env_runner.os")
@mock.patch("ai_traineree.runners.multiagent_env_runner.MultiAgentType")
@mock.patch("ai_traineree.runners.multiagent_env_runner.MultiAgentTaskType")
def test_multiagent_cycle_env_runner_load_state(mock_task, mock_agent, mock_os):
    # Assign
    env_runner = MultiAgentCycleEnvRunner(mock_task, mock_agent, max_iterations=10)
    mock_os.listdir.return_value = [
        "saved_state_e10.json",
        "saved_state_e999.json",
        "other.file",
    ]
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
