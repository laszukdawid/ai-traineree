import json
import logging
import numpy as np
import time
import torch.multiprocessing as mp
import os
import sys

from ai_traineree.agents import AgentBase
from ai_traineree.loggers import DataLogger
from ai_traineree.types import ActionType, DoneType, RewardType, StateType, TaskType
from ai_traineree.types import MultiAgentType, MultiAgentTaskType
from ai_traineree.tasks import PettingZooTask
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


FRAMES_PER_SEC = 45
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def save_gif(path, images: List[np.ndarray]) -> None:
    print(f"Saving as a gif to {path}")
    from PIL import Image
    imgs = [Image.fromarray(img[::2, ::2]) for img in images]  # Reduce /4 size; pick w/2 h/2 pix

    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], optimize=True, quality=85)


class MultiAgentEnvRunner:
    """
    MultiAgentEnvRunner has the same purpose as the EnvRunner but specifically for environments that support multiple agents.
    It's expected that the environments are wrapped in a Task which has typical step and act methods.
    The agent can be any agent which *makes sense* as there aren't any checks like whether the output is discrete.

    Typicall run is
    >>> ma_env_runner = MultiAgentEnvRunner(task, agent)
    >>> ma_env_runner.run()

    Example:
        Check [examples/multi_agents](/examples/multi_agents) directory.

    """

    def __init__(self, task: MultiAgentTaskType, multi_agent: MultiAgentType, mode: str ='coop', max_iterations: int=int(1e5), **kwargs):
        """Expects the environment to come as the TaskType and the agent as the MultiAgentBase.

        Parameters:
            task: An OpenAI gym API compatible task.
            multi_agent: An instance which handles interations between multiple agents.
            mode: Type of interaction between agents.
                Currently supported only `coop` which means that the reward is cummulative for all agents.
            max_iterations: How many iterations can one episode have.

        Keyword Arguments:
            window_len (int): Length of the averaging window for average reward.
            data_logger: An instance of Data Logger, e.g. TensorboardLogger.

        """
        self.logger = logging.getLogger("MAEnvRunner")
        self.task = task
        self.multi_agent: MultiAgentType = multi_agent
        self.max_iterations = max_iterations
        self.model_path = f"{task.name}_{multi_agent.name}"
        self.state_dir = 'run_states'

        self.mode = mode
        self.episode = 0
        self.iteration = 0
        self.all_scores: List[List[RewardType]] = []
        self.all_iterations: List[int] = []
        self.window_len = kwargs.get('window_len', 100)
        self.__images = []
        self._actions = []
        self._rewards = []
        self._dones = []

        self.data_logger = kwargs.get("data_logger")
        self.logger.info("DataLogger: %s", str(self.data_logger))

    def __str__(self) -> str:
        return f"EnvRunner<{self.task.name}, {self.multi_agent.name}>"

    def seed(self, seed: int):
        self.mutli_agent.seed(seed)
        self.task.seed(seed)

    def reset(self):
        """Resets the EnvRunner. The task env and the agent are preserved."""
        self.episode = 0
        self.all_scores: List[List[RewardType]] = []
        self.all_iterations = []
        self.scores_window = deque(maxlen=self.window_len)

    def interact_episode(
        self, eps: float=0, max_iterations: Optional[int]=None,
        render: bool=False, render_gif: bool=False, log_interaction_freq: Optional[int]=None
    ) -> Tuple[List[RewardType], int]:
        score: List[RewardType] = [0.]*self.multi_agent.agents_number
        states: List[StateType] = self.task.reset()

        iterations = 0
        max_iterations = max_iterations if max_iterations is not None else self.max_iterations

        # Only gifs require keeping (S, A, R) list
        if render_gif:
            self.__images = []

        while(iterations < max_iterations):
            iterations += 1
            self.iteration += 1
            if render:
                self.task.render("human")
                time.sleep(1./FRAMES_PER_SEC)

            actions: List[ActionType] = self.multi_agent.act(states, eps)
            next_states: List[StateType] = []
            rewards: List[RewardType] = []
            dones: List[DoneType] = []

            for agent_id in range(self.multi_agent.agents_number):
                next_state, reward, done, _ = self.task.step(actions[agent_id], agent_id=agent_id)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                score[agent_id] += float(reward)  # Score is
                # if done:
                #     break
            if render_gif:
                # OpenAI gym still renders the image to the screen even though it shouldn't. Eh.
                img = self.task.render(mode='rgb_array')
                self.__images.append(img)

            self._actions.append((self.iteration, actions))
            self._dones.append((self.iteration, dones))
            self._rewards.append((self.iteration, rewards))

            self.multi_agent.step(states, actions, rewards, next_states, dones)
            if log_interaction_freq is not None and (iterations % log_interaction_freq) == 0:
                self.log_data_interaction()
            states = next_states
            if any(dones):
                break
        return score, iterations

    def run(
        self,
        reward_goal: float=100.0, max_episodes: int=2000,
        eps_start=1.0, eps_end=0.01, eps_decay=0.995,
        log_episode_freq=1, gif_every_episodes: Optional[int]=None,
        checkpoint_every=200, force_new=False,
    ) -> List[List[RewardType]]:
        """
        Evaluates the multi_agent in the environment.
        The evaluation will stop when the agent reaches the `reward_goal` in the averaged last `self.window_len`, or
        when the number of episodes reaches the `max_episodes`.

        To help debugging one can set the `gif_every_episodes` to a positive integer which relates to how often a gif
        of the episode evaluation is written to the disk.

        Every `checkpoint_every` (default: 200) iterations the Runner will store current state of the runner and the agent.
        These states can be used to resume previous run. By default the runner checks whether there is ongoing run for
        the combination of the environment and the agent.
        """
        self.epsilon = eps_start
        self.reset()
        if not force_new:
            self.load_state(self.model_path)

        mean_scores = []
        epsilons = []

        while (self.episode < max_episodes):
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

            if render_gif and len(self.__images):
                gif_path = "gifs/{}_e{}.gif".format(self.model_path, str(self.episode).zfill(len(str(max_episodes))))
                save_gif(gif_path, self.__images)
                self.__images = []

            if mean_scores[-1] >= reward_goal and len(self.scores_window) == self.window_len:
                print(f'Environment solved after {self.episode} episodes!\tAverage Score: {mean_scores[-1]:.2f}')
                self.save_state(self.model_path)
                self.multi_agent.save_state(f'{self.model_path}_agent.net')
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
        episode = kwargs.get('episodes')[-1]
        score = kwargs.get('scores')[-1]
        iteration = kwargs.get('iterations')[-1]
        mean_score = kwargs.get('mean_scores')[-1]
        epsilon = kwargs.get('epsilons')[-1]
        loss = kwargs.get('loss', {})
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
        episodes: List[int] = kwargs.get('episodes', [])
        for episode, epsilon in zip(episodes, kwargs.get('epsilons', [])):
            self.data_logger.log_value("episode/epsilon", epsilon, episode)

        for episode, mean_score in zip(episodes, kwargs.get('mean_scores', [])):
            self.data_logger.log_value("episode/avg_score", mean_score, episode)

        for episode, score in zip(episodes, kwargs.get('scores', [])):
            self.data_logger.log_value("episode/score", score, episode)

        for episode, iteration in zip(episodes, kwargs.get('iterations', [])):
            self.data_logger.log_value("episode/iterations", iteration, episode)

    def log_data_interaction(self, **kwargs):
        if hasattr(self.multi_agent, 'log_metrics'):
            self.multi_agent.log_metrics(self.data_logger, self.iteration)
        else:
            for loss_name, loss_value in kwargs.get('loss', {}).items():
                self.data_logger.log_value(f"loss/{loss_name}", loss_value, self.iteration)

        while(self._actions):
            step, actions = self._actions.pop(0)
            actions = actions if isinstance(actions, Iterable) else [actions]
            self.data_logger.log_values_dict("env/action", {str(i): a for i, a in enumerate(actions)}, step)

        while(self._rewards):
            step, rewards = self._rewards.pop(0)
            rewards = rewards if isinstance(rewards, Iterable) else [rewards]
            self.data_logger.log_values_dict("env/reward", {str(i): r for i, r in enumerate(rewards)}, step)

        while(self._dones):
            step, dones = self._dones.pop(0)
            dones = dones if isinstance(dones, Iterable) else [dones]
            self.data_logger.log_values_dict("env/done", {str(i): d for i, d in enumerate(dones)}, step)

    def save_state(self, state_name: str):
        """Saves the current state of the runner and the multi_agent.

        Files are stored with appended episode number.
        Agents are saved with their internal saving mechanism.
        """
        state = {
            'tot_iterations': sum(self.all_iterations),
            'episode': self.episode,
            'epsilon': self.epsilon,
            'score': self.all_scores[-1],
            'average_score': sum(self.scores_window) / len(self.scores_window),
            'loss': self.multi_agent.loss,
        }

        Path(self.state_dir).mkdir(parents=True, exist_ok=True)
        self.multi_agent.save_state(f'{self.state_dir}/{state_name}_e{self.episode}.agent')
        with open(f'{self.state_dir}/{state_name}_e{self.episode}.json', 'w') as f:
            json.dump(state, f)

    def load_state(self, state_prefix: str):
        """
        Loads state with the highest episode value for given agent and environment.
        """
        try:
            state_files = list(filter(lambda f: f.startswith(state_prefix) and f.endswith('json'), os.listdir(self.state_dir)))
            e = max([int(f[f.index('_e')+2:f.index('.')]) for f in state_files])
        except Exception:
            self.logger.warning("Couldn't load state. Forcing restart.")
            return

        state_name = [n for n in state_files if n.endswith(f"_e{e}.json")][0][:-5]
        self.logger.info("Loading saved state under: %s/%s.json", self.state_dir, state_name)
        with open(f'{self.state_dir}/{state_name}.json', 'r') as f:
            state = json.load(f)
        self.episode = state.get('episode')
        self.all_scores.append(state.get('score'))
        self.all_iterations = []

        avg_score = state.get('average_score')
        for _ in range(min(self.window_len, self.episode)):
            self.scores_window.append(avg_score)

        self.logger.info("Loading saved agent state: %s/%s.agent", self.state_dir, state_name)
        self.multi_agent.load_state(f'{self.state_dir}/{state_name}.agent')
        self.multi_agent.loss = state.get('loss', 0)


class MultiAgentCycleEnvRunner:
    """
    MultiAgentCycleEnvRunner has the same purpose as the EnvRunner but specifically for environments that support multiple agents.
    It's expected that the environments are wrapped in a Task which has typical step and act methods.
    The agent can be any agent which *makes sense* as there aren't any checks like whether the output is discrete.

    Typicall run is
    >>> ma_env_runner = MultiAgentCycleEnvRunner(task, agent)
    >>> ma_env_runner.run()

    Example:
        Check [examples/multi_agents](/examples/multi_agents) directory.

    """

    def __init__(self, task: PettingZooTask, multi_agent: MultiAgentType, mode: str ='coop', max_iterations: int=int(1e5), **kwargs):
        """Expects the environment to come as the TaskType and the agent as the MultiAgentBase.

        Parameters:
            task: An OpenAI gym API compatible task.
            multi_agent: An instance which handles interations between multiple agents.
            mode: Type of interaction between agents.
                Currently supported only `coop` which means that the reward is cummulative for all agents.
            max_iterations: How many iterations can one episode have.

        Keyword Arguments:
            window_len (int): Length of the averaging window for average reward.
            data_logger: An instance of Data Logger, e.g. TensorboardLogger.

        """
        self.logger = logging.getLogger("MAEnvRunner")
        self.task: PettingZooTask = task
        self.multi_agent: MultiAgentType = multi_agent
        self.max_iterations = max_iterations
        self.model_path = f"{task.name}_{multi_agent.name}"
        self.state_dir = 'run_states'

        self.mode = mode
        self.episode = 0
        self.iteration = 0
        self.all_scores: List[List[RewardType]] = []
        self.all_iterations: List[int] = []
        self.window_len = kwargs.get('window_len', 100)
        self.__images = []
        self._actions = []
        self._rewards = []
        self._dones = []

        self.data_logger = kwargs.get("data_logger")
        self.logger.info("DataLogger: %s", str(self.data_logger))

    def __str__(self) -> str:
        return f"EnvRunner<{self.task.name}, {self.multi_agent.name}>"

    def seed(self, seed: int):
        self.mutli_agent.seed(seed)
        self.task.seed(seed)

    def reset(self):
        """Resets the EnvRunner. The task env and the agent are preserved."""
        self.episode = 0
        self.all_scores: List[List[RewardType]] = []
        self.all_iterations = []
        self.scores_window = deque(maxlen=self.window_len)

    def interact_episode(
        self, eps: float=0, max_iterations: Optional[int]=None,
        render: bool=False, render_gif: bool=False, log_interaction_freq: Optional[int]=None
    ) -> Tuple[Dict[str, RewardType], int]:
        score = defaultdict(int)

        self.task.reset()

        iterations = 0
        max_iterations = max_iterations if max_iterations is not None else self.max_iterations

        # Only gifs require keeping (S, A, R) list
        if render_gif:
            self.__images = []

        while(iterations < max_iterations):
            iterations += 1
            self.iteration += 1
            if render:
                self.task.render("human")
                time.sleep(1./FRAMES_PER_SEC)

            next_states: Dict[str, StateType] = {}
            rewards: Dict[str, RewardType] = {}
            dones: Dict[str, DoneType] = {}

            # for agent_id, agent in range(self.multi_agent.agents):
            for agent_name in self.task.agent_iter(max_iter=self.multi_agent.num_agents):
                state, reward, done, info = self.task.last(agent_name)
                action = self.multi_agent.act(agent_name, state, eps)
                next_state, reward, done, _ = self.task.step(action)

                self.multi_agent.step(agent_name, state, action, reward, next_state, done)

                next_states[agent_name] = next_state
                rewards[agent_name] = reward
                dones[agent_name] = done
                score[agent_name] += float(reward)  # Score is

            if render_gif:
                # OpenAI gym still renders the image to the screen even though it shouldn't. Eh.
                img = self.task.render(mode='rgb_array')
                self.__images.append(img)

            # self._actions.append((self.iteration, actions))
            # self._dones.append((self.iteration, dones))
            # self._rewards.append((self.iteration, rewards))

                if log_interaction_freq is not None and (iterations % log_interaction_freq) == 0:
                    self.log_data_interaction()
            if any(dones.values()):
                break
        return score, iterations

    def run(
        self,
        reward_goal: float=100.0, max_episodes: int=2000,
        eps_start=1.0, eps_end=0.01, eps_decay=0.995,
        log_episode_freq=1, gif_every_episodes: Optional[int]=None,
        checkpoint_every=200, force_new=False,
    ) -> List[List[RewardType]]:
        """
        Evaluates the multi_agent in the environment.
        The evaluation will stop when the agent reaches the `reward_goal` in the averaged last `self.window_len`, or
        when the number of episodes reaches the `max_episodes`.

        To help debugging one can set the `gif_every_episodes` to a positive integer which relates to how often a gif
        of the episode evaluation is written to the disk.

        Every `checkpoint_every` (default: 200) iterations the Runner will store current state of the runner and the agent.
        These states can be used to resume previous run. By default the runner checks whether there is ongoing run for
        the combination of the environment and the agent.
        """
        self.epsilon = eps_start
        self.reset()
        if not force_new:
            self.load_state(self.model_path)

        mean_scores = []
        epsilons = []

        while (self.episode < max_episodes):
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

            if render_gif and len(self.__images):
                gif_path = "gifs/{}_e{}.gif".format(self.model_path, str(self.episode).zfill(len(str(max_episodes))))
                save_gif(gif_path, self.__images)
                self.__images = []

            if mean_scores[-1] >= reward_goal and len(self.scores_window) == self.window_len:
                print(f'Environment solved after {self.episode} episodes!\tAverage Score: {mean_scores[-1]:.2f}')
                self.save_state(self.model_path)
                self.multi_agent.save_state(f'{self.model_path}_agent.net')
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
        episode = kwargs.get('episodes')[-1]
        # score = kwargs.get('scores')[-1]
        iteration = kwargs.get('iterations')[-1]
        mean_score = kwargs.get('mean_scores')[-1]
        epsilon = kwargs.get('epsilons')[-1]
        loss = kwargs.get('loss', {})
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
        episodes: List[int] = kwargs.get('episodes', [])
        for episode, epsilon in zip(episodes, kwargs.get('epsilons', [])):
            self.data_logger.log_value("episode/epsilon", epsilon, episode)

        for episode, mean_score in zip(episodes, kwargs.get('mean_scores', [])):
            self.data_logger.log_value("episode/avg_score", mean_score, episode)

        for episode, score in zip(episodes, kwargs.get('scores', [])):
            self.data_logger.log_value("episode/score", score, episode)

        for episode, iteration in zip(episodes, kwargs.get('iterations', [])):
            self.data_logger.log_value("episode/iterations", iteration, episode)

    def log_data_interaction(self, **kwargs):
        if hasattr(self.multi_agent, 'log_metrics'):
            self.multi_agent.log_metrics(self.data_logger, self.iteration)
        else:
            for loss_name, loss_value in kwargs.get('loss', {}).items():
                self.data_logger.log_value(f"loss/{loss_name}", loss_value, self.iteration)

        while(self._actions):
            step, actions = self._actions.pop(0)
            actions = actions if isinstance(actions, Iterable) else [actions]
            self.data_logger.log_values_dict("env/action", {str(i): a for i, a in enumerate(actions)}, step)

        while(self._rewards):
            step, rewards = self._rewards.pop(0)
            rewards = rewards if isinstance(rewards, Iterable) else [rewards]
            self.data_logger.log_values_dict("env/reward", {str(i): r for i, r in enumerate(rewards)}, step)

        while(self._dones):
            step, dones = self._dones.pop(0)
            dones = dones if isinstance(dones, Iterable) else [dones]
            self.data_logger.log_values_dict("env/done", {str(i): d for i, d in enumerate(dones)}, step)

    def save_state(self, state_name: str):
        """Saves the current state of the runner and the multi_agent.

        Files are stored with appended episode number.
        Agents are saved with their internal saving mechanism.
        """
        state = {
            'tot_iterations': sum(self.all_iterations),
            'episode': self.episode,
            'epsilon': self.epsilon,
            'score': self.all_scores[-1],
            'average_score': sum(self.scores_window) / len(self.scores_window),
            'loss': self.multi_agent.loss,
        }

        Path(self.state_dir).mkdir(parents=True, exist_ok=True)
        self.multi_agent.save_state(f'{self.state_dir}/{state_name}_e{self.episode}.agent')
        with open(f'{self.state_dir}/{state_name}_e{self.episode}.json', 'w') as f:
            json.dump(state, f)

    def load_state(self, state_prefix: str):
        """
        Loads state with the highest episode value for given agent and environment.
        """
        try:
            state_files = list(filter(lambda f: f.startswith(state_prefix) and f.endswith('json'), os.listdir(self.state_dir)))
            e = max([int(f[f.index('_e')+2:f.index('.')]) for f in state_files])
        except Exception:
            self.logger.warning("Couldn't load state. Forcing restart.")
            return

        state_name = [n for n in state_files if n.endswith(f"_e{e}.json")][0][:-5]
        self.logger.info("Loading saved state under: %s/%s.json", self.state_dir, state_name)
        with open(f'{self.state_dir}/{state_name}.json', 'r') as f:
            state = json.load(f)
        self.episode = state.get('episode')
        self.all_scores.append(state.get('score'))
        self.all_iterations = []

        avg_score = state.get('average_score')
        for _ in range(min(self.window_len, self.episode)):
            self.scores_window.append(avg_score)

        self.logger.info("Loading saved agent state: %s/%s.agent", self.state_dir, state_name)
        self.multi_agent.load_state(f'{self.state_dir}/{state_name}.agent')
        self.multi_agent.loss = state.get('loss', 0)
