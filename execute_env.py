import matplotlib
matplotlib.use('TkAgg')

import time

import numpy as np
import pylab as plt
import torch
import gym

from collections import deque

from agents.dqn import Agent
from envs.cart import interact_env


env_name = 'CartPole-v0'
env = gym.make(env_name)

def run_env(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    t0 = time.time()
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    i_episode = 0
    for _ in range(1, n_episodes+1):
        i_episode += 1
        score: int = interact_env(env, agent, eps, False)# render)

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=100.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), f'{env_name}.pth')
            break
    t1 = time.time()
    dt = t1 - t0
    print(f"Training took: {dt} s\tTime per episode: {dt/i_episode}")
    return scores

agent = Agent(env)

interact_env(env, agent, 0, render=True)
scores = run_env(10000, eps_end=0.002, eps_decay=0.9999)
interact_env(env, agent, 0, render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()


