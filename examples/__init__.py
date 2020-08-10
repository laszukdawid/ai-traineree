import numpy as np
from ai_traineree.types import AgentType, RewardType, TaskType
import time

from collections import deque
from typing import Tuple


def interact_episode(task: TaskType, agent: AgentType, eps: float, render=False, max_t=1000) -> Tuple[RewardType, int]:
    score = 0
    state = task.reset()
    iterations = 0
    for _ in range(max_t):
        iterations += 1
        state = np.array(state, np.float32)
        if render:
            task.render()
            time.sleep(0.02)
        action = agent.act(state, eps)
        action = np.array(action, dtype=np.float32)
        next_state, reward, done, _ = task.step(action)
        score += reward
        agent.step(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    return score, iterations


def run_env(
    task: TaskType, agent: AgentType, reward_goal: float=100.0,
    print_every=10,
    max_episodes: int=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995
):
    t0 = time.time()
    all_scores = []
    all_iterations = []
    window_len = 50
    scores_window = deque(maxlen=window_len)
    eps = eps_start

    for episode in range(1, max_episodes+1):
        score, iterations = interact_episode(task, agent, eps)

        scores_window.append(score)
        all_iterations.append(iterations)
        all_scores.append(score)

        mean_score: float = sum(scores_window) / len(scores_window)

        eps = max(eps_end, eps_decay * eps)
        print(f"\rEpisode {episode};\tIter: {iterations};", end="\t")
        print(f"Average Score: {mean_score:.2f};", end='\t')
        if 'critic_loss' in agent.__dict__:
            print(f"Actor loss: {agent.actor_loss:10.4f};", end='\t')
            print(f"Critic loss: {agent.critic_loss:10.4f};", end='\t')
        else:
            print(f"Loss: {agent.last_loss:10.4f};", end='\t')
        print(f"eps: {eps:5.3f}", end="")
        if episode % print_every == 0:
            print()

        if mean_score >= reward_goal:
            print('\nEnvironment solved after {i_episode} episodes!\tAverage Score: {mean_score:.2f}')
            break
    dt = time.time() - t0
    print(f"Training took: {dt} s\tTime per episode: {dt/len(all_scores)}")
    return all_scores
