from ai_traineree.tasks import GymTask
import numpy as np
import pylab as plt
import gym

from ai_traineree.types import TaskType
from ai_traineree.agents.ddpg import DDPGAgent as DDPG
from examples import interact_episode, run_env

import matplotlib
matplotlib.use('TkAgg')


# env_name = 'LunarLander-v2'
env_name = 'LunarLanderContinuous-v2'
env = gym.make(env_name)

task: TaskType = GymTask(env, env_name)
config = {'batch_size': 32}
agent = DDPG(task.state_size, task.action_size, hidden_layers=(400, 300), noise_scale=0.4, config=config)

# interact_episode(task, agent, 0, render=True)
scores = run_env(task, agent, 50, 2000, eps_start=1.0, eps_end=0.05, eps_decay=0.9999)
interact_episode(task, agent, 0, render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
