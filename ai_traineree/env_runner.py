import logging
import numpy as np
import time
import os
import sys
from ai_traineree.types import AgentType, RewardType, TaskType

from collections import deque
from functools import wraps
from pathlib import Path
from typing import List, Optional, Tuple

FRAMES_PER_SEC = 25
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="")


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        t_init = time.perf_counter()
        result = f(*args, **kw)
        t_fin = time.perf_counter()
        print(f'func: {f.__name__} args:[{kw}] took: {t_fin-t_init:2.4f} sec')
        return result
    return wrap


def save_gif(path, images: List[Tuple]) -> None:
    print(f"Saving as a gif to {path}")
    from PIL import Image
    imgs = [Image.fromarray(img) for img in images]

    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    imgs[0].save(path, save_all=True, append_images=imgs[1:])


class EnvRunner:
    """
    EnvRunner, shorter for Environment Runner, is meant to be used as module that runs your experiments.
    It's expected that the environments are wrapped in a Task which has typical step and act methods.
    The agent can be any agent which *makes sense* as there aren't any checks like whether the output is discrete.

    Typicall run is
    >>> env_runner = EnvRunner(task, agent)
    >>> env_runner.run()
    """

    def __init__(self, task: TaskType, agent: AgentType, max_iterations: int=1000, **kwargs):
        """
        Expects the environment to come as the TaskType and the agent as the AgentType.
        Additional args:

        writer: Tensorboard writer.
        """
        self.logger = logging.getLogger("EnvRunner")
        self.task = task
        self.agent = agent
        self.max_iterations = max_iterations
        self.model_path = f"{task.name}_{agent.name}"

        self.all_scores = []
        self.all_iterations = []
        self.window_len = kwargs.get('window_len', 50)
        self.__images = []

        self.writer = kwargs.get("writer")
        self.logger.info("writer: %s", str(self.writer))

    def __str__(self) -> str:
        return f"EnvRunner<{self.task.name}, {self.agent.name}>"

    def reset(self):
        """Resets the EnvRunner. The task env and the agent are preserved."""
        self.all_scores = []
        self.all_iterations = []

    def interact_episode(self, eps: float=0, max_iterations=None, render=False, render_gif=False) -> Tuple[RewardType, int]:
        score = 0
        state = self.task.reset()
        iterations = 0
        max_iterations = max_iterations if max_iterations is not None else self.max_iterations

        # Only gifs require keeping (S, A, R) list
        if render_gif:
            self.__images = []

        while(iterations < max_iterations):
            iterations += 1
            state = np.array(state, np.float32)
            if render:
                self.task.render()
                time.sleep(1./FRAMES_PER_SEC)
            action = self.agent.act(state, eps)
            action = np.array(action, dtype=np.float32)
            next_state, reward, done, _ = self.task.step(action)
            score += reward
            if render_gif:
                # OpenAI gym still renders the image to the screen even though it shouldn't. Eh.
                img = self.task.render(mode='rgb_array')
                self.__images.append(img)
            self.agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        return score, iterations

    @timing
    def run(
        self,
        reward_goal: float=100.0, max_episodes: int=2000,
        eps_start=1.0, eps_end=0.01, eps_decay=0.995, log_every=10,
        gif_every_episodes: Optional[int]=None,
    ):
        """
        Evaluates the agent in the environment.
        The evaluation will stop when the agent reaches the `reward_goal` in the averaged last `self.window_len`, or
        when the number of episodes reaches the `max_episodes`.

        To help debugging one can set the `gif_every_episodes` to a positive integer which relates to how often a gif
        of the episode evaluation is written to the disk.
        """
        self.reset()
        scores_window = deque(maxlen=self.window_len)
        eps = eps_start

        for episode in range(1, max_episodes+1):
            render_gif = gif_every_episodes is not None and (episode % gif_every_episodes) == 0
            score, iterations = self.interact_episode(eps, render_gif=render_gif)

            scores_window.append(score)
            self.all_iterations.append(iterations)
            self.all_scores.append(score)

            mean_score: float = sum(scores_window) / len(scores_window)

            eps = max(eps_end, eps_decay * eps)

            if episode % log_every == 0:
                if 'critic_loss' in self.agent.__dict__:
                    loss = {'actor_loss': self.agent.actor_loss, 'critic_loss': self.agent.critic_loss}
                else:
                    loss = {'loss': self.agent.loss}
                self.info(episode=episode, iterations=iterations, score=score, mean_score=mean_score, epsilon=eps, **loss)

            if render_gif and len(self.__images):
                gif_path = "gifs/{}_e{}.gif".format(self.model_path, str(episode).zfill(len(str(max_episodes))))
                save_gif(gif_path, self.__images)
                self.__images = []

            if mean_score >= reward_goal:
                print(f'Environment solved after {episode} episodes!\tAverage Score: {mean_score:.2f}')
                self.agent.save_state(self.model_path)
                break

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
        line_chunks = ["Episode {episode};", "Iter: {iterations};"]
        line_chunks += ["Current Score: {score:.2f};"]
        line_chunks += ["Average Score: {mean_score:.2f};"]
        if 'critic_loss' in self.agent.__dict__:
            line_chunks += ["Actor loss: {actor_loss:10.4f};"]
            line_chunks += ["Critic loss: {critic_loss:10.4f};"]
        else:
            line_chunks += ["Loss: {loss:10.4f};"]
        line_chunks += ["Epsilon: {epsilon:5.3f};"]
        line = "\t".join(line_chunks)
        self.logger.info(line.format(**kwargs))

    def log_writer(self, **kwargs):
        episode = kwargs['episode']
        self.writer.add_scalar("score", kwargs['score'], episode)
        self.writer.add_scalar("avg_score", kwargs['mean_score'], episode)
        if 'critic_loss' in self.agent.__dict__:
            self.writer.add_scalar("Actor loss", kwargs['actor_loss'], episode)
            self.writer.add_scalar("Critic loss", kwargs['critic_loss'], episode)
        else:
            self.writer.add_scalar("loss", kwargs['loss'], episode)
        self.writer.add_scalar("epsilon", kwargs['epsilon'], episode)
