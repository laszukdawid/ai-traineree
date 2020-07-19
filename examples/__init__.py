import numpy as np
from ai_traineree.types import AgentType, RewardType, TaskType
import time
import torch

from collections import deque
from typing import Iterable


def interact_episode(task: TaskType, agent: AgentType, eps: float, render=False, max_t=1000) -> RewardType:
    score = 0
    state = task.reset()
    for _ in range(max_t):
        if render:
            task.render()
            time.sleep(0.02)
        action = agent.act(state, eps)
        if isinstance(action, Iterable):
            action = np.array(action, dtype=np.float32)
        next_state, reward, done, _ = task.step(action)
        score += reward
        agent.step(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    return score


def run_env(task: TaskType, agent: AgentType, reward_goal: float=100.0, max_episodes: int=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    t0 = time.time()
    all_scores = []
    window_len = 50
    scores_window = deque(maxlen=window_len)
    eps = eps_start

    for episode in range(1, max_episodes+1):
        score = interact_episode(task, agent, eps)

        scores_window.append(score)
        all_scores.append(score)

        mean_score: float = sum(scores_window) / len(scores_window)

        eps = max(eps_end, eps_decay * eps)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, mean_score), end="")
        if episode % 10 == 0:
            agent_loss = agent.last_loss
            print(f'\rEpisode {episode}\tAverage Score: {mean_score:.2f};\tLoss: {agent_loss:10.4f}; eps: {eps:5.3f}')

        if mean_score >= reward_goal:
            print('\nEnvironment solved after {i_episode} episodes!\tAverage Score: {mean_score:.2f}')
            break
    dt = time.time() - t0
    print(f"Training took: {dt} s\tTime per episode: {dt/len(all_scores)}")
    return all_scores
