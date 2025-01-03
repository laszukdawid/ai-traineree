import json
import logging
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch.multiprocessing as mp

from ai_traineree.agents import AgentBase
from ai_traineree.loggers import DataLogger
from ai_traineree.types import TaskType
from ai_traineree.types.experience import Experience

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CMD_RESET = "RESET"
CMD_STOP = "STOP"


class MultiSyncEnvRunner:
    """Execute multiple environments/tasks concurrently with sync steps.

    All environments are distributed to separate processes. The MultiSyncEnvRunner
    acts as a manager that sends data between processes.

    Currently this class only supports training one agent at a time. The agent
    is expected handle stepping multiple steps at a time.
    """

    logger = logging.getLogger("MultiSyncEnvRunner")

    def __init__(self, tasks: list[TaskType], agent: AgentBase, max_iterations: int = int(1e5), **kwargs):
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
        self.logger.setLevel(kwargs.get("logger_level", logging.INFO))

        self.episode = 0
        self.iteration = 0
        self.all_scores = []
        self.all_iterations = []
        self.window_len = kwargs.get("window_len", 100)
        self.scores_window = deque(maxlen=self.window_len)

        self.data_logger: DataLogger | None = kwargs.get("data_logger")
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

        # Infinite loop should be Ok because at the beginning of the loop
        # we wait for new data and at the end we send back.
        while True:
            received = conn.recv()

            if received == CMD_STOP:
                conn.close()
                return

            if received == CMD_RESET:
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

    def init_processes(self) -> None:
        "Initiate agents in processes and establish communication"
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
        checkpoint_every: int | None = 200,
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
            self.init_processes()

            return self._run(
                reward_goal=reward_goal,
                max_episodes=max_episodes,
                max_iterations=max_iterations,
                eps_start=eps_start,
                eps_end=eps_end,
                eps_decay=eps_decay,
                log_episode_freq=log_episode_freq,
                checkpoint_every=checkpoint_every,
                force_new=force_new,
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
        checkpoint_every: int | None = 200,
        force_new: bool = False,
    ):
        # Initiate variables
        self.reset()
        self.epsilon = eps_start
        if not force_new:
            self.load_state(file_prefix=self.model_path)

        _scores = np.zeros(self.task_num)
        _iterations = np.zeros(self.task_num)

        obs = np.empty((len(self.tasks),) + self.tasks[0].obs_space.shape, dtype=np.float32)
        next_obs = obs.copy()
        actions = np.empty((len(self.tasks),) + self.tasks[0].action_space.shape, dtype=np.float32)

        # Start processes just before using them
        for process in self.processes:
            process.start()

        for idx, conn in enumerate(self.parent_conns):
            conn.send(CMD_RESET)
            reset_state = conn.recv()
            obs[idx] = reset_state

        mean_scores = []
        epsilons = []
        mean_score = -float("inf")

        while self.episode < max_episodes:
            self.iteration += 1

            experience = Experience(obs=obs)
            experience = self.agent.act(experience, self.epsilon)
            # TODO: Logic seems Ok.
            # This is likely the problem! Probably agent doesn't understand how to process multi agent data
            actions = experience.action if self.task_num != 1 else [experience.action]
            assert isinstance(actions, list), "For many agents needs to be list"
            experience.update(action=actions)

            # Step all tasks
            self._step_all_tasks(obs, actions)

            # Collect new observations from all stepped tasks
            new_experience, iterations = self._collect_all_tasks()
            experience = experience + new_experience
            next_obs = experience.next_obs
            _scores += experience.reward
            _iterations = iterations

            # All recently evaluated SARS are passed at the same time
            self.agent.step(experience)

            # Training part
            for idx in range(self.task_num):
                # Update Episode number if any agent is DONE or enough ITERATIONS
                if not (experience.done[idx] or _iterations[idx] >= max_iterations):
                    continue

                self.parent_conns[idx].send(CMD_RESET)
                next_obs[idx] = self.parent_conns[idx].recv()

                self.scores_window.append(_scores[idx])
                self.all_scores.append(_scores[idx])
                _scores[idx] = 0

                self.all_iterations.append(_iterations[idx])

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

            obs = next_obs

            self.epsilon = max(eps_end, eps_decay * self.epsilon)

            if mean_score >= reward_goal and len(self.scores_window) == self.window_len:
                print(f"Environment solved after {self.episode} episodes!\tAverage Score: {mean_score:.2f}")
                self.save_state(self.model_path)
                self.agent.save_state(f"{self.model_path}_agent.net")
                break
        return self.all_scores

    def _step_all_tasks(self, obs, actions):
        for t_idx in range(self.task_num):
            self.parent_conns[t_idx].send((t_idx, obs[t_idx], actions[t_idx]))

    def _collect_all_tasks(self) -> tuple[Experience, np.ndarray]:
        obs = np.empty((len(self.tasks),) + self.tasks[0].obs_space.shape, dtype=np.float32)
        next_obs = obs.copy()
        actions = np.empty((len(self.tasks),) + self.tasks[0].action_space.shape, dtype=np.float32)
        dones = np.empty(len(self.tasks))
        rewards = np.empty(len(self.tasks))
        iterations = np.empty(len(self.tasks))

        for t_idx in range(self.task_num):
            obj = self.parent_conns[t_idx].recv()

            idx = obj["idx"]
            rewards[idx] = obj["reward"]
            obs[idx] = obj["state"]
            actions[idx] = obj["action"]
            next_obs[idx] = obj["next_state"]
            dones[idx] = obj["done"]

            iterations[idx] = obj["iteration"]

        experience = Experience(obs=obs, next_obs=next_obs, action=actions, done=dones, reward=rewards)
        return experience, iterations

    def close_all(self):
        while self.parent_conns:
            conn = self.parent_conns.pop(0)
            conn.send(CMD_STOP)
            conn.close()

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
        """Uses data_logger, e.g. Tensorboard, to store env metrics."""
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
