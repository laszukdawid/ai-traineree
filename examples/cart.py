import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pylab as plt
import gym

from ai_traineree.types import TaskType
from ai_traineree.agents.dqn import Agent as DQN
from . import interact_episode, run_env


env_name = 'CartPole-v1'
env = gym.make(env_name)

class Task(TaskType):
    def __init__(self, env, can_render=True):
        self.name = env_name
        self.env = env
        self.can_render = can_render
        self.state_size = sum(env.observation_space.shape)
        self.action_size = env.action_space.n
    
    def reset(self):
        return self.env.reset()
    
    def render(self):
        if self.can_render:
            self.env.render()
        else:
            print("Can't render. Sorry.")  # Yes, this is for haha

    def step(self, action):
        return self.env.step(action)

task = Task(env)
agent = DQN(task)

scores = run_env(task, agent, 40, 5000, eps_end=0.002, eps_decay=0.999)
interact_episode(task, agent, 0, render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()

