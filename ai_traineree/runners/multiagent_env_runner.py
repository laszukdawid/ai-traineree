import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

from ai_traineree.loggers import DataLogger
from ai_traineree.tasks import PettingZooTask
from ai_traineree.types import ActionType, DoneType, MultiAgentTaskType, MultiAgentType, RewardType, StateType
from ai_traineree.types.experience import Experience
from ai_traineree.utils import save_gif

FRAMES_PER_SEC = 45

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


class MultiAgentEnvRunner:
    """
    MultiAgentEnvRunner has the same purpose as the EnvRunner but specifically for environments that support multiple
    agents. It's expected that the environments are wrapped in a Task which has typical step and act methods.
    The agent can be any agent which *makes sense* as there aren't any checks like whether the output is discrete.

    Example:
        >>> ma_env_runner = MultiAgentEnvRunner(task, agent)
        >>> ma_env_runner.run()

    """

    def __init__(
        self,
        task: MultiAgentTaskType,
        multi_agent: MultiAgentType,
        mode: str = "coop",
        max_iterations: int = int(1e5),
        **kwargs,
    ):
        """Expects the environment to come as the TaskType and the agent as the MultiAgentBase.

        Parameters:
            task: An OpenAI gym API compatible task.
            multi_agent: An instance which handles interations between multiple agents.
            mode: Type of interaction between agents.
                Currently supported only `coop` which means that the reward is cumulative for all agents.
            max_iterations: How many iterations can one episode have.

        Keyword Arguments:
            window_len (int): Length of the averaging window for average reward.
            data_logger: An instance of Data Logger, e.g. TensorboardLogger.
            state_dir (str):  Dir path where states are stored. Default: `run_states`.
            debug_log (bool): Whether to produce extensive logging. Default: False.

        """
        self.logger = logging.getLogger("MAEnvRunner")
        self.task = task
        self.multi_agent: MultiAgentType = multi_agent
        self.max_iterations = max_iterations
        self.model_path = f"{task.name}_{multi_agent.model}"
        self.state_dir = kwargs.get("state_dir", "run_states")
        self.window_len = kwargs.get("window_len", 100)

        self.mode = mode
        self.episode = 0
        self.iteration = 0
        self.all_scores: list[list[RewardType]] = []
        self.all_iterations: list[int] = []
        self.scores_window = deque(maxlen=self.window_len)
        self._images = []
        self._debug_log = kwargs.get("debug_log", False)
        self._actions = []
        self._rewards = []
        self._dones = []

        self.data_logger: DataLogger | None = kwargs.get("data_logger")
        self.logger.info("DataLogger: %s", str(self.data_logger))

    def __str__(self) -> str:
        return f"EnvRunner<{self.task.name}, {self.multi_agent.model}>"

    def seed(self, seed: int):
        self.multi_agent.seed(seed)
        self.task.seed(seed)

    def reset(self):
        """Resets the EnvRunner. The task env and the agent are preserved."""
        self.episode = 0
        self.all_scores: list[list[RewardType]] = []
        self.all_iterations = []
        self.scores_window = deque(maxlen=self.window_len)

    def interact_episode(
        self,
        eps: float = 0,
        max_iterations: int | None = None,
        render: bool = False,
        render_gif: bool = False,
        log_interaction_freq: int | None = None,
    ) -> tuple[list[RewardType], int]:
        score: list[RewardType] = [0.0] * self.multi_agent.num_agents
        states: list[StateType] = self.task.reset()

        iterations = 0
        max_iterations = max_iterations if max_iterations is not None else self.max_iterations

        # Only gifs require keeping (S, A, R) list
        if render_gif:
            self._images = []

        while iterations < max_iterations:
            iterations += 1
            self.iteration += 1
            if render:
                self.task.render("human")
                time.sleep(1.0 / FRAMES_PER_SEC)

            next_states: list[StateType] = []
            rewards: list[RewardType] = []
            dones: list[DoneType] = []
            actions: list[ActionType] = []
            for agent_id in range(self.multi_agent.num_agents):
                experience = Experience(obs=states[agent_id])
                experience = self.multi_agent.act(str(agent_id), experience, eps)
                actions.append(experience.action)

                next_state, reward, done, _ = self.task.step(experience.action, agent_id=agent_id)
                experience.update(next_obs=next_state, reward=reward, done=done)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                score[agent_id] += float(reward)  # Score is
                # if done:
                #     break
                self.multi_agent.step(str(agent_id), experience)

            if render_gif:
                # OpenAI gym still renders the image to the screen even though it shouldn't. Eh.
                img = self.task.render(mode="rgb_array")
                self._images.append(img)

            if self._debug_log:
                self._actions.append((self.iteration, actions))
                self._dones.append((self.iteration, dones))
                self._rewards.append((self.iteration, rewards))

            if log_interaction_freq is not None and (iterations % log_interaction_freq) == 0:
                self.log_data_interaction()
            states = next_states
            if any(dones):
                break
        return score, iterations

    def run(
        self,
        reward_goal: float = 100.0,
        max_episodes: int = 2000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        log_episode_freq=1,
        gif_every_episodes: int | None = None,
        checkpoint_every=200,
        force_new=False,
    ) -> list[list[RewardType]]:
        """
        Evaluates the multi_agent in the environment.
        The evaluation will stop when the agent reaches the `reward_goal` in the averaged last `self.window_len`, or
        when the number of episodes reaches the `max_episodes`.

        To help debugging one can set the `gif_every_episodes` to a positive integer which relates to how often a gif
        of the episode evaluation is written to the disk.

        Every `checkpoint_every` (default: 200) iterations the Runner will store current state of the runner and
        the agent. These states can be used to resume previous run. By default the runner checks whether there is
        an ongoing run for the combination of the environment and the agent.
        """
        self.epsilon = eps_start
        self.reset()
        if not force_new:
            self.load_state(self.model_path)

        mean_scores = []
        epsilons = []

        while self.episode < max_episodes:
            self.episode += 1
            render_gif = gif_every_episodes is not None and (self.episode % gif_every_episodes) == 0
            scores, iterations = self.interact_episode(self.epsilon, render_gif=render_gif)

            # TODO: Assumes that score for the episode is a sum of all. That might be Ok with coop but not in general.
            score = sum(scores)

            self.scores_window.append(score)
            self.all_iterations.append(iterations)
            self.all_scores.append(scores)

            mean_scores.append(sum(self.scores_window) / len(self.scores_window))
            epsilons.append(self.epsilon)

            self.epsilon = max(eps_end, eps_decay * self.epsilon)

            if self.episode % log_episode_freq == 0:
                last_episodes = [self.episode - i for i in range(log_episode_freq)[::-1]]
                self.info(
                    episodes=last_episodes,
                    iterations=self.all_iterations[-log_episode_freq:],
                    scores=self.all_scores[-log_episode_freq:],
                    mean_score=mean_scores[-log_episode_freq:],
                    epsilon=epsilons[-log_episode_freq:],
                    loss=self.multi_agent.loss,
                )

            if render_gif and len(self._images):
                gif_path = "gifs/{}_e{}.gif".format(self.model_path, str(self.episode).zfill(len(str(max_episodes))))
                save_gif(gif_path, self._images)
                self._images = []

            if mean_scores[-1] >= reward_goal and len(self.scores_window) == self.window_len:
                print(f"Environment solved after {self.episode} episodes!\tAverage Score: {mean_scores[-1]:.2f}")
                self.save_state(self.model_path)
                self.multi_agent.save_state(f"{self.model_path}_agent.net")
                break

            if self.episode % checkpoint_every == 0:
                self.save_state(self.model_path)

        return self.all_scores

    def info(self, **kwargs):
        """
        Writes out current state into provided loggers.
        Writting to stdout is done through Python's logger, whereas all metrics
        are supposed to be handled via DataLogger. Currently supported are Tensorboard
        and Neptune (neptune.ai). To use one of these `data_logger` is expected.
        """
        if self.data_logger is not None:
            self.log_episode_metrics(**kwargs)
            self.log_data_interaction(**kwargs)
        if self.logger is not None:
            self.log_logger(**kwargs)

    def log_logger(self, **kwargs):
        """Writes out env logs via logger (either stdout or a file)."""
        episode = kwargs.get("episodes")[-1]
        score = kwargs.get("scores")[-1]
        iteration = kwargs.get("iterations")[-1]
        mean_score = kwargs.get("mean_scores")[-1]
        epsilon = kwargs.get("epsilons")[-1]
        loss = kwargs.get("loss", {})
        line_chunks = [f"Episode {episode};"]
        line_chunks += [f"Iter: {iteration};"]
        line_chunks += [f"Current Score: {score:.2f};"]
        line_chunks += [f"Average Score: {mean_score:.2f};"]
        line_chunks += [f"{loss_name.capitalize()}: {loss_value:10.4f}" for (loss_name, loss_value) in loss.items()]
        line_chunks += [f"Epsilon: {epsilon:5.3f};"]
        line = "\t".join(line_chunks)
        try:
            self.logger.info(line.format(**kwargs))
        except Exception:
            print("Line: ", line)
            print("kwargs: ", kwargs)

    def log_episode_metrics(self, **kwargs):
        """Uses data_logger, e.g. Tensorboard, to store env metrics."""
        assert self.data_logger, "Cannot log without DataLogger"
        episodes: list[int] = kwargs.get("episodes", [])
        for episode, epsilon in zip(episodes, kwargs.get("epsilons", [])):
            self.data_logger.log_value("episode/epsilon", epsilon, episode)

        for episode, mean_score in zip(episodes, kwargs.get("mean_scores", [])):
            self.data_logger.log_value("episode/avg_score", mean_score, episode)

        for episode, score in zip(episodes, kwargs.get("scores", [])):
            self.data_logger.log_value("episode/score", score, episode)

        for episode, iteration in zip(episodes, kwargs.get("iterations", [])):
            self.data_logger.log_value("episode/iterations", iteration, episode)

    def log_data_interaction(self, **kwargs):
        if self.data_logger is None:
            self.logger.warning("`log_data_interaction` has no effect without `data_logger`")
            return

        if hasattr(self.multi_agent, "log_metrics"):
            self.multi_agent.log_metrics(self.data_logger, self.iteration)
        else:
            for loss_name, loss_value in kwargs.get("loss", {}).items():
                self.data_logger.log_value(f"loss/{loss_name}", loss_value, self.iteration)

        if self._debug_log:
            while self._actions:
                step, actions = self._actions.pop(0)
                actions = actions if isinstance(actions, Iterable) else [actions]
                self.data_logger.log_values_dict("env/action", {str(i): a for i, a in enumerate(actions)}, step)

            while self._rewards:
                step, rewards = self._rewards.pop(0)
                rewards = rewards if isinstance(rewards, Iterable) else [rewards]
                self.data_logger.log_values_dict("env/reward", {str(i): r for i, r in enumerate(rewards)}, step)

            while self._dones:
                step, dones = self._dones.pop(0)
                dones = dones if isinstance(dones, Iterable) else [dones]
                self.data_logger.log_values_dict("env/done", {str(i): d for i, d in enumerate(dones)}, step)

    def save_state(self, state_name: str):
        """Saves the current state of the runner and the multi_agent.

        Files are stored with appended episode number.
        Agents are saved with their internal saving mechanism.
        """
        state = {
            "tot_iterations": sum(self.all_iterations),
            "episode": self.episode,
            "epsilon": self.epsilon,
            "score": self.all_scores[-1],
            "average_score": sum(self.scores_window) / len(self.scores_window),
            "loss": self.multi_agent.loss,
        }

        Path(self.state_dir).mkdir(parents=True, exist_ok=True)
        self.multi_agent.save_state(f"{self.state_dir}/{state_name}_e{self.episode}.agent")
        with open(f"{self.state_dir}/{state_name}_e{self.episode}.json", "w") as f:
            json.dump(state, f)

    def load_state(self, state_prefix: str):
        """
        Loads state with the highest episode value for given agent and environment.
        """
        try:
            state_files = list(
                filter(lambda f: f.startswith(state_prefix) and f.endswith("json"), os.listdir(self.state_dir))
            )
            e = max([int(f[f.index("_e") + 2 : f.index(".")]) for f in state_files])
        except Exception:
            self.logger.warning("Couldn't load state. Forcing restart.")
            return

        state_name = [n for n in state_files if n.endswith(f"_e{e}.json")][0][:-5]
        self.logger.info("Loading saved state under: %s/%s.json", self.state_dir, state_name)
        with open(f"{self.state_dir}/{state_name}.json", "r") as f:
            state = json.load(f)
        self.episode = state.get("episode")
        self.all_scores.append(state.get("score"))
        self.all_iterations = []

        avg_score = state.get("average_score")
        for _ in range(min(self.window_len, self.episode)):
            self.scores_window.append(avg_score)

        self.logger.info("Loading saved agent state: %s/%s.agent", self.state_dir, state_name)
        self.multi_agent.load_state(f"{self.state_dir}/{state_name}.agent")
        self.multi_agent.loss = state.get("loss", 0)


class MultiAgentCycleEnvRunner:
    """
    MultiAgentCycleEnvRunner has the same purpose as the EnvRunner but specifically for environments that
    support multiple agents. It's expected that the environments are wrapped in a Task which has typical step
    and act methods. The agent can be any agent which *makes sense* as there aren't any checks like whether
    the output is discrete.

    Examples:
        >>> ma_env_runner = MultiAgentCycleEnvRunner(task, agent)
        >>> ma_env_runner.run()

    """

    logger = logging.getLogger("MAEnvRunner")

    def __init__(
        self,
        task: PettingZooTask,
        multi_agent: MultiAgentType,
        mode: str = "coop",
        max_iterations: int = int(1e5),
        **kwargs,
    ):
        """Expects the environment to come as the TaskType and the agent as the MultiAgentBase.

        Parameters:
            task: An OpenAI gym API compatible task.
            multi_agent: An instance which handles interations between multiple agents.
            mode: Type of interaction between agents.
                Currently supported only `coop` which means that the reward is cumulative for all agents.
            max_iterations: How many iterations can one episode have.

        Keyword arguments:

        Keyword Arguments:
            window_len (int): Length of the averaging window for average reward. Default: 100.
            data_logger: An instance of Data Logger, e.g. TensorboardLogger. Default: None.
            state_dir (str):  Dir path where states are stored. Default: `run_states`.
            debug_log (bool): Whether to produce extensive logging. Default: False.

        """
        self.task: PettingZooTask = task
        self.multi_agent: MultiAgentType = multi_agent
        self.max_iterations = max_iterations
        self.model_path = f"{task.name}_{multi_agent.model}"
        self.state_dir = kwargs.get("state_dir", "run_states")

        self.mode = mode
        self.episode: float = 0
        self.iteration = 0
        self.all_scores: list[dict[str, RewardType]] = []
        self.all_iterations: list[int] = []
        self.window_len = kwargs.get("window_len", 100)
        self.scores_window = deque(maxlen=self.window_len)

        self._debug_log = kwargs.get("debug_log", False)
        self._actions = defaultdict(list)
        self._rewards = defaultdict(list)
        self._dones = defaultdict(list)
        self._images = []

        self.data_logger = kwargs.get("data_logger")
        self.logger.info("DataLogger: %s", str(self.data_logger))

    def __str__(self) -> str:
        return f"MultiAgentCycleEnvRunner<{self.task.name}, {self.multi_agent.model}>"

    def seed(self, seed: int) -> None:
        """Sets provided seed to multi agent and task."""
        self.multi_agent.seed(seed)
        self.task.seed(seed)

    def reset(self) -> None:
        """Resets instance. Preserves everything about task and agent."""
        self.episode: float = 0
        self.all_scores: list[dict[str, RewardType]] = []
        self.all_iterations = []
        self.scores_window = deque(maxlen=self.window_len)

    def interact_episode(
        self,
        eps: float = 0,
        max_iterations: int | None = None,
        render: bool = False,
        render_gif: bool = False,
        log_interaction_freq: int | None = None,
    ) -> tuple[dict[str, RewardType], int]:
        score = defaultdict(float)
        iterations = 0
        max_iterations = max_iterations if max_iterations is not None else self.max_iterations
        self.task.reset()

        # Only gifs require keeping (S, A, R) list
        if render_gif:
            self._images = []

        while iterations < max_iterations:
            iterations += 1
            self.iteration += 1
            if render:
                self.task.render("human")
                time.sleep(1.0 / FRAMES_PER_SEC)

            # next_states: dict[str, StateType] = {}
            # rewards: dict[str, RewardType] = {}
            dones: dict[str, DoneType] = {}

            # TODO: Iterate over distinc agents in a single cycle. This `for` doesn't guarantee that.
            for agent_name in self.task.agent_iter(max_iter=self.multi_agent.num_agents):
                state, reward, done, info = self.task.last(agent_name)
                experience = Experience(obs=state)
                experience = self.multi_agent.act(agent_name, experience, eps)
                assert experience.action is not None, "Need to have action after acting"
                next_obs, reward, done, _ = self.task.step(experience.action)

                experience.update(next_obs=next_obs, reward=reward, done=done)
                self.multi_agent.step(agent_name, experience)

                # next_states[agent_name] = next_state
                # rewards[agent_name] = reward
                dones[agent_name] = done
                score[agent_name] += float(reward)

                if self._debug_log:
                    self._actions[agent_name].append((self.iteration, experience.action))
                    self._dones[agent_name].append((self.iteration, experience.done))
                    self._rewards[agent_name].append((self.iteration, experience.reward))

            # Commit last transitions and learn if it's the time
            self.multi_agent.commit()

            if render_gif:
                # OpenAI gym still renders the image to the screen even though it shouldn't. Eh.
                img = self.task.render(mode="rgb_array")
                self._images.append(img)

            if log_interaction_freq is not None and (iterations % log_interaction_freq) == 0:
                self.log_data_interaction()

            if any(dones.values()):
                break

        return score, iterations

    def run(
        self,
        reward_goal: float = 100.0,
        max_episodes: int = 2000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        log_episode_freq=1,
        gif_every_episodes: int | None = None,
        checkpoint_every=200,
        force_new=False,
    ) -> list[dict[str, RewardType]]:
        """
        Evaluates the Multi Agent in the environment.
        The evaluation will stop when the agent reaches the `reward_goal` in the averaged last `self.window_len`, or
        when the number of episodes reaches the `max_episodes`.

        To help debugging one can set the `gif_every_episodes` to a positive integer which relates to how often a gif
        of the episode evaluation is written to the disk.

        Every `checkpoint_every` (default: 200) iterations the Runner will store current state of the runner and
        the agent. These states can be used to resume previous run. By default the runner checks whether there is
        an ongoing run for the combination of the environment and the agent.
        """
        self.epsilon = eps_start
        self.reset()
        if not force_new:
            self.load_state(self.model_path)

        mean_scores = []
        epsilons = []

        while self.episode < max_episodes:
            self.episode += 1
            render_gif = gif_every_episodes is not None and (self.episode % gif_every_episodes) == 0
            scores, iterations = self.interact_episode(self.epsilon, render_gif=render_gif)

            # TODO: Assumes that score for the episode is a sum of all. That might be Ok with coop but not in general.
            score = sum(scores.values())

            self.scores_window.append(score)
            self.all_iterations.append(iterations)
            self.all_scores.append(scores)

            mean_scores.append(sum(self.scores_window) / len(self.scores_window))
            epsilons.append(self.epsilon)

            self.epsilon = max(eps_end, eps_decay * self.epsilon)

            if self.episode % log_episode_freq == 0:
                last_episodes = [self.episode - i for i in range(log_episode_freq)[::-1]]
                self.info(
                    episodes=last_episodes,
                    iterations=self.all_iterations[-log_episode_freq:],
                    scores=self.all_scores[-log_episode_freq:],
                    mean_scores=mean_scores[-log_episode_freq:],
                    epsilons=epsilons[-log_episode_freq:],
                    loss=self.multi_agent.loss,
                )

            if render_gif and len(self._images):
                gif_path = "gifs/{}_e{}.gif".format(self.model_path, str(self.episode).zfill(len(str(max_episodes))))
                save_gif(gif_path, self._images)
                self._images = []

            if mean_scores[-1] >= reward_goal and len(self.scores_window) == self.window_len:
                print(f"Environment solved after {self.episode} episodes!\tAverage Score: {mean_scores[-1]:.2f}")
                self.save_state(self.model_path)
                self.multi_agent.save_state(f"{self.model_path}_agent.net")
                break

            if self.episode % checkpoint_every == 0:
                self.save_state(self.model_path)

        return self.all_scores

    def info(self, **kwargs):
        """
        Writes out current state into provided loggers.
        Writting to stdout is done through Python's logger, whereas all metrics
        are supposed to be handled via DataLogger. Currently supported are Tensorboard
        and Neptune (neptune.ai). To use one of these `data_logger` is expected.
        """
        if self.data_logger is not None:
            self.log_episode_metrics(**kwargs)
            self.log_data_interaction(**kwargs)
        if self.logger is not None:
            self.log_logger(**kwargs)

    def log_logger(self, **kwargs):
        """Writes out env logs via logger (either stdout or a file)."""
        episode = kwargs.get("episodes")[-1]
        # score = kwargs.get('scores')[-1]
        iteration = kwargs.get("iterations")[-1]
        mean_score = kwargs.get("mean_scores")[-1]
        epsilon = kwargs.get("epsilons")[-1]
        loss = kwargs.get("loss", {})
        line_chunks = [f"Episode {episode};"]
        line_chunks += [f"Iter: {iteration};"]
        # line_chunks += [f"Current Score: {score:.2f};"]
        line_chunks += [f"Average Score: {mean_score:.2f};"]
        line_chunks += [f"{loss_name.capitalize()}: {loss_value:10.4f}" for (loss_name, loss_value) in loss.items()]
        line_chunks += [f"Epsilon: {epsilon:5.3f};"]
        line = "\t".join(line_chunks)
        try:
            self.logger.info(line.format(**kwargs))
        except Exception:
            print("Line: ", line)
            print("kwargs: ", kwargs)

    def log_episode_metrics(self, **kwargs):
        """Uses data_logger, e.g. Tensorboard, to store env metrics."""
        assert self.data_logger, "Cannot log without DataLogger"
        episodes: list[int] = kwargs.get("episodes", [])
        for episode, epsilon in zip(episodes, kwargs.get("epsilons", [])):
            self.data_logger.log_value("episode/epsilon", epsilon, episode)

        for episode, mean_score in zip(episodes, kwargs.get("mean_scores", [])):
            self.data_logger.log_value("episode/avg_score", mean_score, episode)

        for episode, score in zip(episodes, kwargs.get("scores", [])):
            self.data_logger.log_values_dict("episode/score", score, episode)

        for episode, iteration in zip(episodes, kwargs.get("iterations", [])):
            self.data_logger.log_value("episode/iterations", iteration, episode)

    def log_data_interaction(self, **kwargs):
        if self.data_logger is None:
            self.logger.warning("`log_data_interaction` has no effect without `data_logger`")
            return

        if hasattr(self.multi_agent, "log_metrics"):
            self.multi_agent.log_metrics(self.data_logger, self.iteration, full_log=kwargs.get("full_log", False))
        else:
            for loss_name, loss_value in kwargs.get("loss", {}).items():
                self.data_logger.log_value(f"loss/{loss_name}", loss_value, self.iteration)

        if self._debug_log:
            for agent_name in self.multi_agent.agent_names:
                while self._actions[agent_name]:
                    step, actions = self._actions[agent_name].pop(0)
                    actions = actions if isinstance(actions, Iterable) else [actions]
                    self.data_logger.log_values_dict(
                        f"{agent_name}/action", {str(i): a for i, a in enumerate(actions)}, step
                    )

                while self._rewards[agent_name]:
                    step, rewards = self._rewards[agent_name].pop(0)
                    rewards = rewards if isinstance(rewards, Iterable) else [rewards]
                    self.data_logger.log_values_dict(
                        f"{agent_name}/reward", {str(i): r for i, r in enumerate(rewards)}, step
                    )

                while self._dones[agent_name]:
                    step, dones = self._dones[agent_name].pop(0)
                    dones = dones if isinstance(dones, Iterable) else [dones]
                    self.data_logger.log_values_dict(
                        f"{agent_name}/done", {str(i): d for i, d in enumerate(dones)}, step
                    )

    def save_state(self, state_name: str):
        """Saves the current state of the runner and the multi_agent.

        Files are stored with appended episode number.
        Agents are saved with their internal saving mechanism.
        """
        state = {
            "tot_iterations": sum(self.all_iterations),
            "episode": self.episode,
            "epsilon": self.epsilon,
            "score": self.all_scores[-1],
            "average_score": sum(self.scores_window) / len(self.scores_window),
            "loss": self.multi_agent.loss,
        }

        Path(self.state_dir).mkdir(parents=True, exist_ok=True)
        self.multi_agent.save_state(f"{self.state_dir}/{state_name}_e{self.episode}.agent")
        with open(f"{self.state_dir}/{state_name}_e{self.episode}.json", "w") as f:
            json.dump(state, f)

    def load_state(self, file_prefix: str):
        """
        Loads state with the highest episode value for given agent and environment.
        """
        try:
            state_files = list(
                filter(lambda f: f.startswith(file_prefix) and f.endswith("json"), os.listdir(self.state_dir))
            )
            e = max([int(f[f.index("_e") + 2 : f.index(".")]) for f in state_files])
            state_name = [n for n in state_files if n.endswith(f"_e{e}.json")][0][:-5]
        except Exception:
            self.logger.warning("Couldn't load state. Forcing restart.")
            return

        self.logger.info("Loading saved state under: %s/%s.json", self.state_dir, state_name)
        with open(f"{self.state_dir}/{state_name}.json", "r") as f:
            state = json.load(f)
        self.episode = state.get("episode")
        self.epsilon = state.get("epsilon")

        self.all_scores.append(state.get("score"))
        self.all_iterations = []

        avg_score = state.get("average_score")
        for _ in range(min(self.window_len, self.episode)):
            self.scores_window.append(avg_score)

        self.logger.info("Loading saved agent state: %s/%s.agent", self.state_dir, state_name)
        self.multi_agent.load_state(f"{self.state_dir}/{state_name}.agent")
        self.multi_agent.loss = state.get("loss", 0)
