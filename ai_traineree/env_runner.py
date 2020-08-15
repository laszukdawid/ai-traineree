import logging
import numpy as np
import time
import sys
from ai_traineree.types import AgentType, RewardType, TaskType

from collections import deque
from functools import wraps
from typing import Tuple

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


class EnvRunner:

    def __init__(self, task: TaskType, agent: AgentType, max_iterations: int=1000, window_len: int=50):
        self.logger = logging.getLogger("EnvRunner")
        self.task = task
        self.agent = agent
        self.max_iterations = max_iterations
        self.model_path = f"./{task.name}_{agent.name}"

        self.window_len = window_len

    def __str__(self) -> str:
        return f"EnvRunner<{self.task.name}, {self.agent.name}>"

    def reset(self):
        self.all_scores = []

    def interact_episode(self, eps: float=0, render=False, max_iterations=None) -> Tuple[RewardType, int]:
        score = 0
        state = self.task.reset()
        iterations = 0
        max_iterations = max_iterations if max_iterations is not None else self.max_iterations
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
            self.agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        return score, iterations

    @timing
    def run(
        self,
        reward_goal: float=100.0, max_episodes: int=2000,
        eps_start=1.0, eps_end=0.01, eps_decay=0.995, print_every=10,
    ):
        all_scores = []
        all_iterations = []
        scores_window = deque(maxlen=self.window_len)
        eps = eps_start

        for episode in range(1, max_episodes+1):
            score, iterations = self.interact_episode(eps)

            scores_window.append(score)
            all_iterations.append(iterations)
            all_scores.append(score)

            mean_score: float = sum(scores_window) / len(scores_window)

            eps = max(eps_end, eps_decay * eps)

            if episode % print_every == 0:
                line_chunks = [f"Episode {episode};", f"Iter: {iterations};"]
                line_chunks += [f"Current Score: {score:.2f};"]
                line_chunks += [f"Average Score: {mean_score:.2f};"]
                if 'critic_loss' in self.agent.__dict__:
                    line_chunks += [f"Actor loss: {self.agent.actor_loss:10.4f};"]
                    line_chunks += [f"Critic loss: {self.agent.critic_loss:10.4f};"]
                else:
                    line_chunks += [f"Loss: {self.agent.last_loss:10.4f};"]
                line_chunks += [f"Epsilon: {eps:5.3f};"]
                line = "\t".join(line_chunks)
                self.logger.info(line)

            if mean_score >= reward_goal:
                print(f'Environment solved after {episode} episodes!\tAverage Score: {mean_score:.2f}')
                self.agent.save_state(self.model_path)
                break

        return all_scores
