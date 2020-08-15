from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.env_runner import EnvRunner
from ai_traineree.tasks import GymTask

import numpy as np
import pylab as plt
import gym


env_name = 'CartPole-v1'
env = gym.make(env_name)

task = GymTask(env, env_name)
agent = DQNAgent(task.state_size, task.action_size)
env_runner = EnvRunner(task, agent)

scores = env_runner.run(reward_goal=100, max_episodes=5000, eps_end=0.002, eps_decay=0.999)
env_runner.interact_episode(100, render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
