from ai_traineree.types import AgentType, TaskType
import time
import torch

from collections import deque


def interact_episode(task, agent, eps, render=False, max_t=1000) -> int:
    score = 0
    state = task.reset()
    for _ in range(max_t):
        if render:
            task.render()
            time.sleep(0.05)
        action = agent.act(state, eps)
        next_state, reward, done, _ = task.step(action)
        score += reward
        agent.step(state, action, score, next_state, done)
        state = next_state
        if done:
            break
    return score

def run_env(task: TaskType, agent: AgentType, reward_goal: float=100.0, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):

    t0 = time.time()
    all_scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    i_episode = 0
    for _ in range(1, n_episodes+1):
        i_episode += 1
        score: int = interact_episode(task, agent, eps, False)

        scores_window.append(score)
        all_scores.append(score)

        mean_score: float = sum(scores_window)/len(scores_window)

        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score))

        if mean_score >= reward_goal:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, mean_score))
            # torch.save(agent.qnetwork_local.state_dict(), f'{task.name}.pth')
            torch.save(agent.describe_agent(), f'{task.name}.pth')
            break
    t1 = time.time()
    dt = t1 - t0
    print(f"Training took: {dt} s\tTime per episode: {dt/i_episode}")
    return all_scores

