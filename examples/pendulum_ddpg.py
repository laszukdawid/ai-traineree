import numpy as np
import pylab as plt
import gym

from ai_traineree.agents.ddpg import DDPGAgent
from ai_traineree.tasks import GymTask
from examples import EnvRunner


env_name = 'Pendulum-v0'
env = gym.make(env_name)

task = GymTask(env, env_name)
config = {'batch_size': 32}
agent = DDPGAgent(task.state_size, task.action_size, hidden_layers=(400, 300), noise_scale=1., clip=(-2, 2), config=config)
env_runner = EnvRunner(task, agent)

env_runner.interact_episode(0, render=True)
scores = env_runner.run(0, 2000, eps_start=1.0, eps_end=0.05, eps_decay=0.999)
env_runner.interact_episode(0, render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
