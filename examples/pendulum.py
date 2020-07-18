import numpy as np
import pylab as plt
import gym

from ai_traineree.types import TaskType
from ai_traineree.agents.ddpg import DDPGAgent
from ai_traineree.agents.dqn import Agent as DQN
from examples import interact_episode, run_env

import matplotlib
matplotlib.use('TkAgg')


class Task(TaskType):
    def __init__(self, env, can_render=True):
        self.name = env_name
        self.env = env
        self.can_render = can_render
        self.state_size = 3
        self.action_size = 1

    def reset(self):
        return self.env.reset()

    def render(self):
        if self.can_render:
            self.env.render()
        else:
            print("Can't render. Sorry.")  # Yes, this is for haha

    def step(self, actions):
        return self.env.step(actions)


env_name = 'Pendulum-v0'
env = gym.make(env_name)


task = Task(env)
config = {'batch_size': 32}
agent = DDPGAgent(task.state_size, task.action_size, hidden_layers=(400, 300), noise_scale=1., clip=(-2, 2), config=config)

# interact_episode(task, agent, 0, render=True)
scores = run_env(task, agent, 0, 2000, eps_start=1.0, eps_end=0.05, eps_decay=0.999)
interact_episode(task, agent, 0, render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
