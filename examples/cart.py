from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.tasks import GymTask
from examples import interact_episode, run_env

import numpy as np
import pylab as plt
import gym


env_name = 'CartPole-v1'
env = gym.make(env_name)

task = GymTask(env, env_name)
agent = DQNAgent(task.state_size, task.action_size)

scores = run_env(task, agent, 100, 5000, eps_end=0.002, eps_decay=0.999)
interact_episode(task, agent, 100, render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
