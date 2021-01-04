import json
import logging
import numpy as np
import time
import torch.multiprocessing as mp
import os
import sys

from ai_traineree.types import ActionType, AgentType, DoneType, MultiAgentType, RewardType, StateType, TaskType
from ai_traineree.types import MultiAgentTaskType
from collections import deque
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple


FRAMES_PER_SEC = 25
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


class EnvRunner:
    """
    EnvRunner, shorter for Environment Runner, is meant to be used as module that runs your experiments.
    It's expected that the environments are wrapped in a Task which has typical step and act methods.
    The agent can be any agent which *makes sense* as there aren't any checks like whether the output is discrete.

    Typicall run is
    >>> env_runner = EnvRunner(task, agent)
    >>> env_runner.run()
    """

    logger = logging.getLogger("EnvRunner")

    def __init__(self, task: TaskType, agent: AgentType, max_iterations: int=int(1e5), **kwargs):
        """
        Expects the environment to come as the TaskType and the agent as the AgentType.

        Keyword parameters:
            window_len (int): Length of the score averaging window.
            writer: Tensorboard writer.
        """
        self.task = task
        self.agent = agent
        self.max_iterations = max_iterations
        self.model_path = f"{task.name}_{agent.name}"
        self.state_dir = 'run_states'

        self.episode = 0
        self.iteration = 0
        self.all_scores = []
        self.all_iterations = []
        self.window_len = kwargs.get('window_len', 100)
        self.scores_window = deque(maxlen=self.window_len)
        self.__images = []

        self.writer = kwargs.get("writer")
        self.logger.info("writer: %s", str(self.writer))
        if self.writer:
            self.writer.add_hparams(self.agent.hparams, {})

        self._actions: List[Any] = []
        self._rewards: List[Any] = []
        self._dones: List[Any] = []

    def __str__(self) -> str:
        return f"EnvRunner<{self.task.name}, {self.agent.name}>"

    def reset(self):
        """Resets the EnvRunner. The task env and the agent are preserved."""
        self.episode = 0
        self.all_scores = []
        self.all_iterations = []
        self.scores_window = deque(maxlen=self.window_len)

    def interact_episode(
        self,
        eps: float=0,
        max_iterations: Optional[int]=None,
        render: bool=False,
        render_gif: bool=False,
        log_interaction_freq: Optional[int]=1,
    ) -> Tuple[RewardType, int]:
        score = 0
        state = self.task.reset()
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

            action = self.agent.act(state, eps)
            self._actions.append((self.iteration, action))

            next_state, reward, done, _ = self.task.step(action)
            self._rewards.append((self.iteration, reward))
            self._dones.append((self.iteration, done))

            score += float(reward)
            if render_gif:
                # OpenAI gym still renders the image to the screen even though it shouldn't. Eh.
                img = self.task.render(mode='rgb_array')
                self.__images.append(img)

            self.agent.step(state, action, reward, next_state, done)

            if log_interaction_freq is not None and (iterations % log_interaction_freq) == 0:
                self.log_interaction()
            state = next_state
            if done:
                break
        return score, iterations

    def run(
        self,
        reward_goal: float=100.0,
        max_episodes: int=2000,
        eps_start: float=1.0,
        eps_end: float=0.01,
        eps_decay: float=0.995,
        log_every: int=10,
        gif_every_episodes: Optional[int]=None,
        checkpoint_every=200,
        force_new: bool=False,
    ) -> List[float]:
        """
        Evaluates the agent in the environment.
        The evaluation will stop when the agent reaches the `reward_goal` in the averaged last `self.window_len`, or
        when the number of episodes reaches the `max_episodes`.

        To help debugging one can set the `gif_every_episodes` to a positive integer which relates to how often a gif
        of the episode evaluation is written to the disk.

        Every `checkpoint_every` (default: 200) iterations the Runner will store current state of the runner and the agent.
        These states can be used to resume previous run. By default the runner checks whether there is ongoing run for
        the combination of the environment and the agent.

        Parameters:
            reward_goal: Goal to achieve on the average reward.
            max_episode: After how many episodes to stop regardless of the score.
            eps_start: Epsilon-greedy starting value.
            eps_end: Epislon-greeedy lowest value.
            eps_decay: Epislon-greedy decay value, eps[i+1] = eps[i] * eps_decay.
            log_every: Number of episodes between state logging.
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
            self.load_state(self.model_path)
        mean_scores = []
        epsilons = []

        while (self.episode < max_episodes):
            self.episode += 1
            render_gif = gif_every_episodes is not None and (self.episode % gif_every_episodes) == 0
            score, iterations = self.interact_episode(self.epsilon, render_gif=render_gif, log_interaction_freq=10)

            self.scores_window.append(score)
            self.all_iterations.append(iterations)
            self.all_scores.append(score)

            mean_scores.append(sum(self.scores_window) / len(self.scores_window))
            epsilons.append(self.epsilon)

            self.epsilon = max(eps_end, eps_decay * self.epsilon)

            if self.episode % log_every == 0:
                last_episodes = [self.episode - i for i in range(log_every)[::-1]]
                self.info(
                    episodes=last_episodes,
                    iterations=self.all_iterations[-log_every:],
                    scores=self.all_scores[-log_every:],
                    mean_scores=mean_scores[-log_every:],
                    epsilons=epsilons[-log_every:],
                    loss=self.agent.loss,
                )

            if render_gif and len(self.__images):
                gif_path = "gifs/{}_e{}.gif".format(self.model_path, str(self.episode).zfill(len(str(max_episodes))))
                save_gif(gif_path, self.__images)
                self.__images = []

            if mean_scores[-1] >= reward_goal and len(self.scores_window) == self.window_len:
                print(f'Environment solved after {self.episode} episodes!\tAverage Score: {mean_scores[-1]:.2f}')
                self.save_state(self.model_path)
                self.agent.save_state(f'{self.model_path}_agent.net')
                break

            if self.episode % checkpoint_every == 0:
                self.save_state(self.model_path)

        # Store hyper parameters and experiment metrics in logger so that it's easier to compare runs
        if self.writer:
            end_metrics = {
                "hparam/total_iterations": sum(self.all_iterations),
                "hparam/total_episodes": len(self.all_iterations),
                "hparam/score": mean_scores[-1],
            }
            self.writer.add_hparams(self.agent.hparams, end_metrics, run_name="hparams")

        return self.all_scores

    def info(self, **kwargs):
        """
        Writes out current state into provided loggers.
        Currently supports stdout logger (default on) and Tensorboard SummaryWriter initiated through EnvRun(writer=...)).
        """
        if self.writer is not None:
            self.log_writer(**kwargs)
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
        self.logger.info(line.format(**kwargs))

    def log_writer(self, **kwargs):
        """Uses writer, e.g. Tensorboard, to store env metrics."""
        episodes: List[int] = kwargs.get('episodes', [])
        for episode, epsilon in zip(episodes, kwargs.get('epsilons', [])):
            self.writer.add_scalar("episode/epsilon", epsilon, episode)

        for episode, mean_score in zip(episodes, kwargs.get('mean_scores', [])):
            self.writer.add_scalar("episode/avg_score", mean_score, episode)

        for episode, score in zip(episodes, kwargs.get('scores', [])):
            self.writer.add_scalar("episode/score", score, episode)

        for episode, iteration in zip(episodes, kwargs.get('iterations', [])):
            self.writer.add_scalar("episode/iterations", iteration, episode)
        self.log_interaction(**kwargs)

    def log_interaction(self, **kwargs):
        if self.writer is None:
            return

        if hasattr(self.agent, 'log_writer'):
            self.agent.log_writer(self.writer, self.iteration)
        else:
            for loss_name, loss_value in kwargs.get('loss', {}).items():
                self.writer.add_scalar(f"loss/{loss_name}", loss_value, self.iteration)

        while(len(self._actions) > 0):
            step, actions = self._actions.pop()
            actions = actions if isinstance(actions, Sequence) else [actions]
            self.writer.add_scalars("env/action", {str(i): a for i, a in enumerate(actions)}, step)

        while(len(self._rewards) > 0):
            step, rewards = self._rewards.pop()
            rewards = rewards if isinstance(rewards, Sequence) else [rewards]
            self.writer.add_scalars("env/reward", {str(i): r for i, r in enumerate(rewards)}, step)

        while(len(self._dones) > 0):
            step, dones = self._dones.pop()
            dones = dones if isinstance(dones, Sequence) else [dones]
            self.writer.add_scalars("env/done", {str(i): d for i, d in enumerate(dones)}, step)

    def save_state(self, state_name: str):
        """Saves the current state of the runner and the agent.

        Files are stored with appended episode number.
        Agents are saved with their internal saving mechanism.
        """
        state = {
            'tot_iterations': sum(self.all_iterations),
            'episode': self.episode,
            'epsilon': self.epsilon,
            'score': self.all_scores[-1],
            'average_score': sum(self.scores_window) / len(self.scores_window),
            'loss': self.agent.loss,
        }

        Path(self.state_dir).mkdir(parents=True, exist_ok=True)
        self.agent.save_state(f'{self.state_dir}/{state_name}_e{self.episode}.agent')
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
        self.epsilon = state.get('epsilon')

        self.all_scores.append(state.get('score'))
        self.all_iterations = []

        avg_score = state.get('average_score')
        for _ in range(min(self.window_len, self.episode)):
            self.scores_window.append(avg_score)

        self.logger.info("Loading saved agent state: %s/%s.agent", self.state_dir, state_name)
        self.agent.load_state(f'{self.state_dir}/{state_name}.agent')
        self.agent.loss = state.get('loss', 0)


class MultiSyncEnvRunner:
    """Execute multiple environments/tasks concurrently with sync steps.

    All environments are distributed to separate processes. The MultiSyncEnvRunner
    acts as a manager that sends data between processes.

    Currently this class only supports training one agent at a time. The agent
    is expected handle stepping multiple steps at a time.
    """

    logger = logging.getLogger("MultiSyncEnvRunner")

    def __init__(self, tasks: List[TaskType], agent: AgentType, max_iterations: int=int(1e5), **kwargs):
        """
        Expects the environment to come as the TaskType and the agent as the AgentType.

        Keyword parameters:
            window_len (int): Length of the score averaging window.
            writer: Tensorboard writer.
        """
        self.tasks = tasks
        self.task_num = len(tasks)
        self.num_processes = int(kwargs.get("processes", len(tasks)))
        self.processes = []
        self.parent_conns = []
        self.child_conns = []

        self.agent = agent
        self.max_iterations = max_iterations
        self.model_path = f"{tasks[0].name}_{agent.name}"
        self.state_dir = 'run_states'

        self.episode = 0
        self.iteration = 0
        self.all_scores = []
        self.all_iterations = []
        self.window_len = kwargs.get('window_len', 100)
        self.scores_window = deque(maxlen=self.window_len)

        self.writer = kwargs.get("writer")
        self.logger.info("writer: %s", str(self.writer))

    def __str__(self) -> str:
        return f"MultiSyncEnvRunner<{[t.name for t in self.tasks]}, {self.agent.name}>"

    def __del__(self):
        try:
            self.close_all()
        except Exception:
            pass

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

            conn.send({
                "idx": t_idx,
                "state": state,
                "action": action,
                "next_state": next_state,
                "reward": reward,
                "done": done,
                "iteration": iteration,
            })

    def init_network(self):
        for p_idx in range(self.num_processes):
            parent_conn, child_conn = mp.Pipe()
            self.parent_conns.append(parent_conn)
            self.child_conns.append(child_conn)

            process = mp.Process(target=self.step_task, args=(child_conn, self.tasks[p_idx]))
            self.processes.append(process)

    def run(
        self,
        reward_goal: float=100.0,
        max_episodes: int=2000,
        max_iterations: int=int(1e6),
        eps_start: float=1.0,
        eps_end: float=0.01,
        eps_decay: float=0.995,
        log_every: int=10,
        checkpoint_every=200,
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
            log_every: Number of episodes between state logging.
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
                reward_goal, max_episodes, max_iterations,
                eps_start, eps_end, eps_decay,
                log_every, checkpoint_every, force_new,
            )

        finally:
            # All connections and processes need to be closed regardless of any exception.
            # Don't let get away. Who knows what zombie process are capable of?!
            self.close_all()

    def _run(
        self,
        reward_goal: float=100.0,
        max_episodes: int=2000,
        max_iterations: int=1000,
        eps_start: float=1.0,
        eps_end: float=0.01,
        eps_decay: float=0.995,
        log_every: int=10,
        checkpoint_every=200,
        force_new=False,
    ):
        self.epsilon = eps_start
        self.reset()
        if not force_new:
            self.load_state(self.model_path)

        scores = np.zeros(self.task_num)
        iterations = np.zeros(self.task_num)

        states = np.empty((self.num_processes, self.tasks[0].state_size), dtype=np.float32)
        next_states = states.copy()
        actions = np.empty((len(self.tasks), self.tasks[0].action_size), dtype=np.float32)
        dones = np.empty(len(self.tasks))
        rewards = np.empty(len(self.tasks))

        # Start processes just before using them
        for process in self.processes:
            process.start()

        for idx, conn in enumerate(self.parent_conns):
            conn.send("RESET")
            reset_state = conn.recv()
            states[idx] = reset_state

        mean_scores = []
        epsilons = []
        mean_score = -float('inf')

        while (self.episode < max_episodes):
            self.iteration += self.task_num
            iterations += 1

            actions = self.agent.act(states, self.epsilon)

            for t_idx in range(self.task_num):
                action = actions[t_idx].cpu().numpy().flatten()
                self.parent_conns[t_idx].send((t_idx, states[t_idx], action))

            for t_idx in range(self.task_num):
                obj = self.parent_conns[t_idx].recv()            

                idx = obj['idx']
                rewards[idx] = obj['reward']
                states[idx] = obj['state']
                actions[idx] = obj['action']
                next_states[idx] = obj['next_state']
                dones[idx] = obj['done']

                iterations[idx] = obj['iteration']
                scores[idx] += obj['reward']

            # All recently evaluated SARS are passed at the same time
            self.agent.step(states, actions, rewards, next_states, dones)

            for idx in range(self.task_num):
                if not (dones[idx] or iterations[idx] > max_iterations):
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
                if self.episode % log_every == 0:
                    last_episodes = [self.episode - i for i in range(log_every)[::-1]]
                    self.info(
                        episodes=last_episodes,
                        iterations=self.all_iterations[-log_every:],
                        scores=self.all_scores[-log_every:],
                        mean_scores=mean_scores[-log_every:],
                        epsilons=epsilons[-log_every:],
                    )

                if self.episode % checkpoint_every == 0:
                    self.save_state(self.model_path)

            states = next_states

            self.epsilon = max(eps_end, eps_decay * self.epsilon)

            if mean_score >= reward_goal and len(self.scores_window) == self.window_len:
                print(f'Environment solved after {self.episode} episodes!\tAverage Score: {mean_score:.2f}')
                self.save_state(self.model_path)
                self.agent.save_state(f'{self.model_path}_agent.net')
                break
        return self.all_scores

    def close_all(self):
        while(self.parent_conns):
            self.parent_conns.pop().close()

        while(self.child_conns):
            self.child_conns.pop().close()

        while(self.processes):
            p = self.processes.pop()
            p.terminate()
            p.join()

    def info(self, **kwargs):
        """
        Writes out current state into provided loggers.
        Currently supports stdout logger (default on) and Tensorboard SummaryWriter initiated through EnvRun(writer=...)).
        """
        if self.writer is not None:
            self.log_writer(**kwargs)
        if self.logger is not None:
            self.log_logger(**kwargs)

    def log_logger(self, **kwargs):
        """Writes out env logs via logger (either stdout or a file)."""
        episode = kwargs.get('episodes')[-1]
        score = kwargs.get('scores')[-1]
        iteration = kwargs.get('iterations')[-1]
        mean_score = kwargs.get('mean_scores')[-1]
        epsilon = kwargs.get('epsilons')[-1]
        line_chunks = [f"Episode {episode};"]
        line_chunks += [f"Iter: {iteration};"]
        line_chunks += [f"Current Score: {score:.2f};"]
        line_chunks += [f"Average Score: {mean_score:.2f};"]
        line_chunks += [f"Epsilon: {epsilon:5.3f};"]
        line = "\t".join(line_chunks)
        self.logger.info(line.format(**kwargs))

    def log_writer(self, **kwargs):
        """Uses writer, e.g. Tensorboard, to store env metrics."""
        episodes: List[int] = kwargs.get('episodes', [])
        for episode, epsilon in zip(episodes, kwargs.get('epsilons', [])):
            self.writer.add_scalar("episode/epsilon", epsilon, episode)

        for episode, mean_score in zip(episodes, kwargs.get('mean_scores', [])):
            self.writer.add_scalar("episode/avg_score", mean_score, episode)

        for episode, score in zip(episodes, kwargs.get('scores', [])):
            self.writer.add_scalar("episode/score", score, episode)

        for episode, iteration in zip(episodes, kwargs.get('iterations', [])):
            self.writer.add_scalar("episode/iterations", iteration, episode)
        self.log_interaction(**kwargs)

    def log_interaction(self, **kwargs):
        if self.writer and hasattr(self.agent, 'log_writer'):
            self.agent.log_writer(self.writer, self.iteration)

    def save_state(self, state_name: str):
        """Saves the current state of the runner and the agent.

        Files are stored with appended episode number.
        Agents are saved with their internal saving mechanism.
        """
        state = {
            'tot_iterations': sum(self.all_iterations),
            'episode': self.episode,
            'epsilon': self.epsilon,
            'score': self.all_scores[-1],
            'average_score': sum(self.scores_window) / len(self.scores_window),
            'loss': self.agent.loss,
        }

        Path(self.state_dir).mkdir(parents=True, exist_ok=True)
        self.agent.save_state(f'{self.state_dir}/{state_name}_e{self.episode}.agent')
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
        self.epsilon = state.get('epsilon')

        self.all_scores.append(state.get('score'))
        self.all_iterations = []

        avg_score = state.get('average_score')
        for _ in range(min(self.window_len, self.episode)):
            self.scores_window.append(avg_score)

        self.logger.info("Loading saved agent state: %s/%s.agent", self.state_dir, state_name)
        self.agent.load_state(f'{self.state_dir}/{state_name}.agent')
        self.agent.loss = state.get('loss', 0)


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
        """Expects the environment to come as the TaskType and the agent as the AgentType.

        Parameters:
            task: An OpenAI gym API compatible task.
            multi_agent: An instance which handles interations between multiple agents.
            mode: Type of interaction between agents.
                Currently supported only `coop` which means that the reward is cummulative for all agents.
            max_iterations: How many iterations can one episode have.

        Keyword Arguments:
            window_len (int): Length of the averaging window for average reward.
            writer: Tensorboard writer.

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

        self.writer = kwargs.get("writer")
        self.logger.info("writer: %s", str(self.writer))

    def __str__(self) -> str:
        return f"EnvRunner<{self.task.name}, {self.multi_agent.name}>"

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
                self.log_interaction()
            states = next_states
            if any(dones):
                break
        return score, iterations

    def run(
        self,
        reward_goal: float=100.0, max_episodes: int=2000,
        eps_start=1.0, eps_end=0.01, eps_decay=0.995,
        log_every=10, gif_every_episodes: Optional[int]=None,
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

            if self.episode % log_every == 0:
                last_episodes = [self.episode - i for i in range(log_every)[::-1]]
                self.info(
                    episodes=last_episodes,
                    iterations=self.all_iterations[-log_every:],
                    scores=self.all_scores[-log_every:],
                    mean_score=mean_scores[-log_every:],
                    epsilon=epsilons[-log_every:],
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
        Currently supports stdout logger (default on) and Tensorboard SummaryWriter initiated through EnvRun(writer=...)).
        """
        if self.writer is not None:
            self.log_writer(**kwargs)
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

    def log_writer(self, **kwargs):
        """Uses writer, e.g. Tensorboard, to store env metrics."""
        episodes: List[int] = kwargs.get('episodes', [])
        for episode, epsilon in zip(episodes, kwargs.get('epsilons', [])):
            self.writer.add_scalar("episode/epsilon", epsilon, episode)

        for episode, mean_score in zip(episodes, kwargs.get('mean_scores', [])):
            self.writer.add_scalar("episode/avg_score", mean_score, episode)

        for episode, score in zip(episodes, kwargs.get('scores', [])):
            self.writer.add_scalar("episode/score", score, episode)

        for episode, iteration in zip(episodes, kwargs.get('iterations', [])):
            self.writer.add_scalar("episode/iterations", iteration, episode)
        self.log_interaction(**kwargs)

    def log_interaction(self, **kwargs):
        if hasattr(self.multi_agent, 'log_writer'):
            self.multi_agent.log_writer(self.writer, self.iteration)
        else:
            for loss_name, loss_value in kwargs.get('loss', {}).items():
                self.writer.add_scalar(f"loss/{loss_name}", loss_value, self.iteration)

        while(len(self._actions) > 0):
            step, actions = self._actions.pop()
            actions = actions if isinstance(actions, Sequence) else [actions]
            self.writer.add_scalars("env/action", {str(i): a for i, a in enumerate(actions)}, step)

        while(len(self._rewards) > 0):
            step, rewards = self._rewards.pop()
            rewards = rewards if isinstance(rewards, Sequence) else [rewards]
            self.writer.add_scalars("env/reward", {str(i): r for i, r in enumerate(rewards)}, step)

        while(len(self._dones) > 0):
            step, dones = self._dones.pop()
            dones = dones if isinstance(dones, Sequence) else [dones]
            self.writer.add_scalars("env/done", {str(i): d for i, d in enumerate(dones)}, step)

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