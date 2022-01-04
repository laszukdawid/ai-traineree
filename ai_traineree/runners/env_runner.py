import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import torch.multiprocessing as mp

from ai_traineree.agents import AgentBase
from ai_traineree.loggers import DataLogger
from ai_traineree.types import RewardType, TaskType
from ai_traineree.types.experience import Experience
from ai_traineree.utils import save_gif

FRAMES_PER_SEC = 45

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


class EnvRunner:
    """
    EnvRunner, short for Environment Runner, is meant to be used as module that runs your experiments.
    It's expected that the environments are wrapped in a Task which has typical step and act methods.
    The agent can be any agent which *makes sense* as there aren't any checks like whether the output is discrete.

    Examples:
        >>> env_runner = EnvRunner(task, agent)
        >>> env_runner.run()
    """

    logger = logging.getLogger("EnvRunner")

    def __init__(self, task: TaskType, agent: AgentBase, max_iterations: int = int(1e5), **kwargs):
        """
        Expects the environment to come as the TaskType and the agent as the AgentBase.

        Keyword Arguments:
            window_len (int): Length of the score averaging window.
            data_logger: An instance of Data Logger, e.g. TensorboardLogger.
            logger_level: Logging level. Default: logging.INFO.

        """
        self.task = task
        self.agent = agent
        self.max_iterations = max_iterations
        self.model_path = f"{task.name}_{agent.model}"
        self.state_dir = "run_states"

        self.episode = 0
        self.iteration = 0
        self.all_scores = []
        self.all_iterations = []
        self.window_len = kwargs.get("window_len", 100)
        self.scores_window = deque(maxlen=self.window_len)
        self.__images = []

        self.logger.setLevel(kwargs.get("logger_level", logging.INFO))
        self.data_logger: Optional[DataLogger] = kwargs.get("data_logger")
        if self.data_logger:
            self.logger.info("DataLogger: %s", str(self.data_logger))
            self.data_logger.set_hparams(self.agent.hparams, {})

        self._debug_log: bool = bool(kwargs.get("debug_log", False))
        self._exp: List[Tuple[int, Experience]] = []
        self._actions: List[Any] = []
        self._states: List[Any] = []
        self._rewards: List[Any] = []
        self._dones: List[Any] = []
        self._noises: List[Any] = []

        self.seed(kwargs.get("seed"))

    def __str__(self) -> str:
        return f"EnvRunner<{self.task.name}, {self.agent.model}>"

    def seed(self, seed):
        if isinstance(seed, (int, float)):
            self.agent.seed(seed)
            self.task.seed(seed)

    def reset(self):
        """Resets the EnvRunner. The task env and the agent are preserved."""
        self.episode = 0
        self.all_scores = []
        self.all_iterations = []
        self.scores_window = deque(maxlen=self.window_len)

    def interact_episode(
        self,
        train: bool = False,
        eps: float = 0,
        max_iterations: Optional[int] = None,
        render: bool = False,
        render_gif: bool = False,
        log_interaction_freq: Optional[int] = 10,
        full_log_interaction_freq: Optional[int] = 1000,
    ) -> Tuple[RewardType, int]:
        score = 0
        obs = self.task.reset()
        iterations = 0
        max_iterations = max_iterations if max_iterations is not None else self.max_iterations
        done = False

        # Only gifs require keeping (S, A, R) list
        if render_gif:
            self.__images = []

        self.agent.train = train

        while iterations < max_iterations and not done:
            iterations += 1
            self.iteration += 1
            if render:
                self.task.render("human")
                time.sleep(1.0 / FRAMES_PER_SEC)

            experience = Experience(obs=obs)
            experience = self.agent.act(experience, eps)
            assert experience.get("action") is not None, "Act method should update action on experience"
            action = experience.action

            next_obs, reward, done, _ = self.task.step(action)
            self._rewards.append((self.iteration, reward))

            if self._debug_log:
                self._exp.append((self.iteration, experience))
                self._actions.append((self.iteration, action))
                self._states.append((self.iteration, obs))
                self._dones.append((self.iteration, done))
                noise = experience.get("noise")
                if noise is not None:
                    self._noises.append((self.iteration, noise))

            score += float(reward)
            if render_gif:
                # OpenAI gym still renders the image to the screen even though it shouldn't. Eh.
                img = self.task.render(mode="rgb_array")
                self.__images.append(img)

            experience.update(action=action, reward=reward, next_obs=next_obs, done=done)
            self.agent.step(experience)

            # Plot interactions every `log_interaction_freq` iterations.
            # Plot full log (including weights) every `full_log_interaction_freq` iterations.
            if (log_interaction_freq and (iterations % log_interaction_freq) == 0) or (
                full_log_interaction_freq and (self.iteration % full_log_interaction_freq) == 0
            ):
                full_log = full_log_interaction_freq and (iterations % full_log_interaction_freq) == 0
                self.log_data_interaction(full_log=full_log)

            # n -> n+1  => S(n) <- S(n+1)
            obs = next_obs

        return score, iterations

    def run(
        self,
        reward_goal: float = 100.0,
        max_episodes: int = 2000,
        test_every: int = 10,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: float = 0.995,
        log_episode_freq: int = 1,
        log_interaction_freq: int = 10,
        gif_every_episodes: Optional[int] = None,
        checkpoint_every: Optional[int] = 200,
        force_new: bool = False,
    ) -> List[float]:
        """
        Evaluates the agent in the environment.
        The evaluation will stop when the agent reaches the `reward_goal` in the averaged last `self.window_len`, or
        when the number of episodes reaches the `max_episodes`.

        To help debugging one can set the `gif_every_episodes` to a positive integer which relates to how often a gif
        of the episode evaluation is written to the disk.

        Every `checkpoint_every` (default: 200) iterations the Runner will store current state of the runner
        and the agent. These states can be used to resume previous run. By default the runner checks whether
        there is ongoing run for the combination of the environment and the agent.

        Parameters:
            reward_goal: Goal to achieve on the average reward.
            max_episode: After how many episodes to stop regardless of the score.
            test_every: Number of episodes between agent test run (without learning). Default: 10.
            eps_start: Epsilon-greedy starting value.
            eps_end: Epislon-greeedy lowest value.
            eps_decay: Epislon-greedy decay value, eps[i+1] = eps[i] * eps_decay.
            log_episode_freq: Number of episodes between state logging.
            gif_every_episodes: Number of episodes between storing last episode as a gif.
            checkpoint_every: Number of episodes between storing the whole state, so that
                in case of failure it can be safely resumed from it.
            force_new: Flag whether to resume from previously stored state (False), or to
                start learning from a clean state (True).

        Returns:
            All obtained scores from all episodes.

        """
        self.epsilon = eps_start
        self.reset()
        if not force_new:
            self.load_state(file_prefix=self.model_path)
        mean_scores = []
        epsilons = []

        while self.episode < max_episodes:
            self.episode += 1
            train_agent = (self.episode % test_every) != 0
            render_gif = gif_every_episodes is not None and (self.episode % gif_every_episodes) == 0
            eps = self.epsilon if train_agent else 0

            score, iterations = self.interact_episode(
                train=train_agent, eps=eps, render_gif=render_gif, log_interaction_freq=log_interaction_freq
            )

            if not train_agent:
                self.scores_window.append(score)
                mean_scores.append(sum(self.scores_window) / len(self.scores_window))

            self.all_iterations.append(iterations)
            self.all_scores.append(score)

            epsilons.append(eps)

            self.epsilon = max(eps_end, eps_decay * self.epsilon)

            if self.episode % log_episode_freq == 0:
                last_episodes = [self.episode - i for i in range(log_episode_freq)[::-1]]
                self.info(
                    train_mode=train_agent,
                    episodes=last_episodes,
                    iterations=self.all_iterations[-log_episode_freq:],
                    scores=self.all_scores[-log_episode_freq:],
                    mean_scores=mean_scores[-log_episode_freq:],
                    epsilons=epsilons[-log_episode_freq:],
                    loss=self.agent.loss,
                )

            if render_gif and len(self.__images):
                gif_path = "gifs/{}_e{}.gif".format(self.model_path, str(self.episode).zfill(len(str(max_episodes))))
                save_gif(gif_path, self.__images)
                self.__images = []

            if len(mean_scores) and mean_scores[-1] >= reward_goal and len(self.scores_window) == self.window_len:
                print(f"Environment solved after {self.episode} episodes!\tAverage Score: {mean_scores[-1]:.2f}")
                self.save_state(self.model_path)
                self.agent.save_state(f"{self.model_path}_agent.net")
                break

            if checkpoint_every is not None and self.episode % checkpoint_every == 0:
                self.save_state(self.model_path)
        # END training

        # Store hyper parameters and experiment metrics in logger so that it's easier to compare runs
        if self.data_logger:
            end_metrics = {
                "hparam/total_iterations": sum(self.all_iterations),
                "hparam/total_episodes": len(self.all_iterations),
                "hparam/score": mean_scores[-1],
            }
            self.data_logger.set_hparams(self.agent.hparams, end_metrics, run_name="hparams")

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
        if kwargs is None:
            return

        line_chunks = []
        episodes = kwargs.get("episodes")
        if episodes is not None:
            is_train = kwargs.get("train_mode", True)
            test_mark = "" if is_train else "T"
            line_chunks += [f"Episode {episodes[-1]}{test_mark}"]

        scores = kwargs.get("scores")
        if scores is not None:
            line_chunks += [f"Current Score: {scores[-1]:.2f}"]

        iterations = kwargs.get("iterations")
        if iterations is not None:
            line_chunks += [f"Iter: {iterations[-1]}"]

        mean_scores = kwargs.get("mean_scores")
        if mean_scores is not None and len(mean_scores):
            line_chunks += [f"Average Score: {mean_scores[-1]:.2f}"]

        epsilons = kwargs.get("epsilons")
        if epsilons is not None:
            line_chunks += [f"Epsilon: {epsilons[-1]:5.3f}"]

        loss = kwargs.get("loss", {})
        if loss is not None:
            line_chunks += [f"{loss_name.capitalize()}: {loss_value:10.4f}" for (loss_name, loss_value) in loss.items()]

        line = ";\t".join(line_chunks)
        self.logger.info(line)

    def log_episode_metrics(self, **kwargs):
        """Uses DataLogger, e.g. TensorboardLogger, to store env metrics."""
        episodes: List[int] = kwargs.get("episodes", [])
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
            return

        if hasattr(self.agent, "log_metrics"):
            self.agent.log_metrics(self.data_logger, self.iteration, full_log=kwargs.get("full_log", False))
        else:
            for loss_name, loss_value in kwargs.get("loss", {}).items():
                self.data_logger.log_value(f"loss/{loss_name}", loss_value, self.iteration)

        while self._debug_log and self._exp:
            step, exp = self._exp.pop(0)
            noise_params = exp.get("noise_params")
            if not noise_params:
                continue
            self.data_logger.log_values_dict("env/noise_params", {str(i): a for i, a in enumerate(noise_params)}, step)

        while self._debug_log and self._states:
            step, states = self._states.pop(0)
            states = states if isinstance(states, Iterable) else [states]
            self.data_logger.log_values_dict("env/states", {str(i): a for i, a in enumerate(states)}, step)

        while self._debug_log and self._actions:
            step, actions = self._actions.pop(0)
            actions = actions if isinstance(actions, Iterable) else [actions]
            self.data_logger.log_values_dict("env/action", {str(i): a for i, a in enumerate(actions)}, step)

        while self._debug_log and self._noises:
            step, noises = self._noises.pop(0)
            noises = noises if isinstance(noises, Iterable) else [noises]
            self.data_logger.log_values_dict("env/noise", {str(i): a for i, a in enumerate(noises)}, step)

        while self._debug_log and self._rewards:
            step, rewards = self._rewards.pop(0)
            self.data_logger.log_value("env/reward", float(rewards), step)

        while self._debug_log and self._dones:
            step, dones = self._dones.pop(0)
            self.data_logger.log_value("env/done", int(dones), step)

    def save_state(self, state_name: str):
        """Saves the current state of the runner and the agent.

        Files are stored with appended episode number.
        Agents are saved with their internal saving mechanism.
        """
        state = {
            "tot_iterations": sum(self.all_iterations),
            "episode": self.episode,
            "epsilon": self.epsilon,
            "score": self.all_scores[-1],
            "average_score": sum(self.scores_window) / len(self.scores_window),
            "loss": self.agent.loss,
        }

        Path(self.state_dir).mkdir(parents=True, exist_ok=True)
        self.agent.save_state(f"{self.state_dir}/{state_name}_e{self.episode}.agent")
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
            recent_episode_num = max([int(f[f.index("_e") + 2 : f.index(".")]) for f in state_files])
            state_name = [n for n in state_files if n.endswith(f"_e{recent_episode_num}.json")][0][:-5]
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
        self.agent.load_state(path=f"{self.state_dir}/{state_name}.agent")
        self.agent.loss = state.get("loss", 0)


class MultiSyncEnvRunner:
    """Execute multiple environments/tasks concurrently with sync steps.

    All environments are distributed to separate processes. The MultiSyncEnvRunner
    acts as a manager that sends data between processes.

    Currently this class only supports training one agent at a time. The agent
    is expected handle stepping multiple steps at a time.
    """

    logger = logging.getLogger("MultiSyncEnvRunner")

    def __init__(self, tasks: List[TaskType], agent: AgentBase, max_iterations: int = int(1e5), **kwargs):
        """
        Expects the environment to come as the TaskType and the agent as the AgentBase.

        Keyword Arguments:
            window_len (int): Length of the score averaging window.
            data_logger: An instance of Data Logger, e.g. TensorboardLogger.
        """
        self.tasks = tasks
        self.task_num = len(tasks)
        self.num_processes = int(kwargs.get("processes", len(tasks)))
        self.processes = []
        self.parent_conns = []
        self.child_conns = []

        self.agent = agent
        self.max_iterations = max_iterations
        self.model_path = f"{tasks[0].name}_{agent.model}"
        self.state_dir = "run_states"

        self.episode = 0
        self.iteration = 0
        self.all_scores = []
        self.all_iterations = []
        self.window_len = kwargs.get("window_len", 100)
        self.scores_window = deque(maxlen=self.window_len)

        self.data_logger: Optional[DataLogger] = kwargs.get("data_logger")
        self.logger.info("DataLogger: %s", str(self.data_logger))

    def __str__(self) -> str:
        return f"MultiSyncEnvRunner<{[t.name for t in self.tasks]}, {self.agent.model}>"

    def __del__(self):
        try:
            self.close_all()
        except Exception:
            self.logger.exception("Exception while clossing all connections")

    def reset(self):
        """Resets the EnvRunner. The task env and the agent are preserved."""
        self.episode = 0
        self.all_scores = []
        self.all_iterations = []
        self.scores_window = deque(maxlen=self.window_len)

    @staticmethod
    def step_task(conn, task):
        iteration = 0
        task.reset()
        while True:
            received = conn.recv()

            if received == "STOP":
                conn.close()
                return

            if received == "RESET":
                conn.send(task.reset())
                iteration = 0
                continue

            t_idx, state, action = received
            iteration += 1
            task_out = task.step(action)
            next_state, reward, done, _ = task_out

            conn.send(
                {
                    "idx": t_idx,
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                    "reward": reward,
                    "done": done,
                    "iteration": iteration,
                }
            )

    def init_network(self):
        for p_idx in range(self.num_processes):
            parent_conn, child_conn = mp.Pipe()
            self.parent_conns.append(parent_conn)
            self.child_conns.append(child_conn)

            process = mp.Process(target=self.step_task, args=(child_conn, self.tasks[p_idx]))
            self.processes.append(process)

    def run(
        self,
        reward_goal: float = 100.0,
        max_episodes: int = 2000,
        max_iterations: int = int(1e6),
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: float = 0.995,
        log_episode_freq: int = 1,
        checkpoint_every: Optional[int] = 200,
        force_new=False,
    ):
        """
        Evaluates the agent in the environment.
        The evaluation will stop when the agent reaches the `reward_goal` in the averaged last `self.window_len`, or
        when the number of episodes reaches the `max_episodes`.

        To help debugging one can set the `gif_every_episodes` to a positive integer which relates to how often a gif
        of the episode evaluation is written to the disk.

        Parameters:
            reward_goal: Goal to achieve on the average reward.
            max_episode: After how many episodes to stop regardless of the score.
            eps_start: Epsilon-greedy starting value.
            eps_end: Epislon-greeedy lowest value.
            eps_decay: Epislon-greedy decay value, eps[i+1] = eps[i] * eps_decay.
            log_episode_freq: Number of episodes between state logging.
            checkpoint_every: Number of episodes between storing the whole state, so that
                in case of failure it can be safely resumed from it.
            force_new: Flag whether to resume from previously stored state (False), or to
                start learning from a clean state (True).

        Returns:
            All obtained scores from all episodes.

        """

        # This method is mainly a wrapper around self._run to make it safer.
        # Somehow better option might be to add decorators but given unsure existance
        # of this class we'll refrain from doing so right now.
        try:
            # Initiate all processes and connections
            self.init_network()

            return self._run(
                reward_goal,
                max_episodes,
                max_iterations,
                eps_start,
                eps_end,
                eps_decay,
                log_episode_freq,
                checkpoint_every,
                force_new,
            )

        finally:
            # All connections and processes need to be closed regardless of any exception.
            # Don't let get away. Who knows what zombie process are capable of?!
            self.close_all()

    def _run(
        self,
        reward_goal: float = 100.0,
        max_episodes: int = 2000,
        max_iterations: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: float = 0.995,
        log_episode_freq: int = 1,
        checkpoint_every: Optional[int] = 200,
        force_new=False,
    ):
        self.epsilon = eps_start
        self.reset()
        if not force_new:
            self.load_state(file_prefix=self.model_path)

        scores = np.zeros(self.task_num)
        iterations = np.zeros(self.task_num)

        observations = np.empty((self.num_processes, self.tasks[0].obs_size), dtype=np.float32)
        next_states = observations.copy()
        actions = np.empty((len(self.tasks), self.tasks[0].action_size), dtype=np.float32)
        dones = np.empty(len(self.tasks))
        rewards = np.empty(len(self.tasks))

        # Start processes just before using them
        for process in self.processes:
            process.start()

        for idx, conn in enumerate(self.parent_conns):
            conn.send("RESET")
            reset_state = conn.recv()
            observations[idx] = reset_state

        mean_scores = []
        epsilons = []
        mean_score = -float("inf")

        while self.episode < max_episodes:
            self.iteration += self.task_num
            iterations += 1

            experience = Experience(obs=observations)
            experience = self.agent.act(experience, self.epsilon)
            actions = experience.action if self.task_num != 1 else [experience.action]
            assert isinstance(actions, list), "For many agents needs to be list"
            experience.update(action=actions)

            for t_idx in range(self.task_num):
                self.parent_conns[t_idx].send((t_idx, observations[t_idx], actions[t_idx]))

            for t_idx in range(self.task_num):
                obj = self.parent_conns[t_idx].recv()

                idx = obj["idx"]
                rewards[idx] = obj["reward"]
                observations[idx] = obj["state"]
                actions[idx] = obj["action"]
                next_states[idx] = obj["next_state"]
                dones[idx] = obj["done"]

                iterations[idx] = obj["iteration"]
                scores[idx] += obj["reward"]

            # All recently evaluated SARS are passed at the same time
            experience.update(reward=rewards, obs=observations, action=actions, next_obs=next_states, done=dones)
            self.agent.step(experience)

            for idx in range(self.task_num):
                if not (dones[idx] or iterations[idx] >= max_iterations):
                    continue

                self.parent_conns[idx].send("RESET")
                next_states[idx] = self.parent_conns[idx].recv()

                self.scores_window.append(scores[idx])
                self.all_scores.append(scores[idx])
                scores[idx] = 0

                self.all_iterations.append(iterations[idx])

                self.episode += 1
                mean_score: float = sum(self.scores_window) / len(self.scores_window)
                mean_scores.append(mean_score)
                epsilons.append(self.epsilon)

                # Log only once per evaluation, and outside s
                if self.episode % log_episode_freq == 0:
                    last_episodes = [self.episode - i for i in range(log_episode_freq)[::-1]]
                    self.info(
                        episodes=last_episodes,
                        iterations=self.all_iterations[-log_episode_freq:],
                        scores=self.all_scores[-log_episode_freq:],
                        mean_scores=mean_scores[-log_episode_freq:],
                        epsilons=epsilons[-log_episode_freq:],
                        loss=self.agent.loss,
                    )

                if checkpoint_every is not None and self.episode % checkpoint_every == 0:
                    self.save_state(self.model_path)

            observations = next_states

            self.epsilon = max(eps_end, eps_decay * self.epsilon)

            if mean_score >= reward_goal and len(self.scores_window) == self.window_len:
                print(f"Environment solved after {self.episode} episodes!\tAverage Score: {mean_score:.2f}")
                self.save_state(self.model_path)
                self.agent.save_state(f"{self.model_path}_agent.net")
                break
        return self.all_scores

    def close_all(self):
        while self.parent_conns:
            self.parent_conns.pop(0).close()

        while self.child_conns:
            self.child_conns.pop(0).close()

        while self.processes:
            p = self.processes.pop(0)
            p.terminate()
            p.join()

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
        loss = kwargs.get("loss", {})
        mean_score = kwargs.get("mean_scores")[-1]
        epsilon = kwargs.get("epsilons")[-1]
        line_chunks = [f"Episode {episode};"]
        line_chunks += [f"Iter: {iteration};"]
        line_chunks += [f"Current Score: {score:.2f};"]
        line_chunks += [f"Average Score: {mean_score:.2f};"]
        line_chunks += [f"{loss_name.capitalize()}: {loss_value:10.4f}" for (loss_name, loss_value) in loss.items()]
        line_chunks += [f"Epsilon: {epsilon:5.3f};"]
        line = "\t".join(line_chunks)
        self.logger.info(line.format(**kwargs))

    def log_episode_metrics(self, **kwargs):
        """Uses data_logger, e.g. Tensorboard, to store env metrics."""
        episodes: List[int] = kwargs.get("episodes", [])
        for episode, epsilon in zip(episodes, kwargs.get("epsilons", [])):
            self.data_logger.log_value("episode/epsilon", epsilon, episode)

        for episode, mean_score in zip(episodes, kwargs.get("mean_scores", [])):
            self.data_logger.log_value("episode/avg_score", mean_score, episode)

        for episode, score in zip(episodes, kwargs.get("scores", [])):
            self.data_logger.log_value("episode/score", score, episode)

        for episode, iteration in zip(episodes, kwargs.get("iterations", [])):
            self.data_logger.log_value("episode/iterations", iteration, episode)

    def log_data_interaction(self, **kwargs):
        if self.data_logger and hasattr(self.agent, "log_metrics"):
            self.agent.log_metrics(self.data_logger, self.iteration, full_log=kwargs.get("full_log", False))

    def save_state(self, state_name: str):
        """Saves the current state of the runner and the agent.

        Files are stored with appended episode number.
        Agents are saved with their internal saving mechanism.
        """
        state = {
            "tot_iterations": sum(self.all_iterations),
            "episode": self.episode,
            "epsilon": self.epsilon,
            "score": self.all_scores[-1],
            "average_score": sum(self.scores_window) / len(self.scores_window),
            "loss": self.agent.loss,
        }

        Path(self.state_dir).mkdir(parents=True, exist_ok=True)
        self.agent.save_state(f"{self.state_dir}/{state_name}_e{self.episode}.agent")
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
            recent_episode_num = max([int(f[f.index("_e") + 2 : f.index(".")]) for f in state_files])
            state_name = [n for n in state_files if n.endswith(f"_e{recent_episode_num}.json")][0][:-5]
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
        self.agent.load_state(f"{self.state_dir}/{state_name}.agent")
        self.agent.loss = state.get("loss", 0)
