from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pylab as plt
import torch

seed = 32167
torch.set_deterministic(True)
torch.manual_seed(seed)
writer = SummaryWriter()

env_name = 'CartPole-v1'
task = GymTask(env_name, seed=seed)
agent = DQNAgent(task.state_size, task.action_size, n_steps=5, seed=seed)
env_runner = EnvRunner(task, agent, writer=writer, seed=seed)

scores = env_runner.run(
    reward_goal=100, max_episodes=300, eps_end=0.002, eps_decay=0.99,
    gif_every_episodes=500, force_new=True,
)
env_runner.interact_episode(render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
