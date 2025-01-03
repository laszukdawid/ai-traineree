import numpy as np
import pylab as plt
import torch

from aitraineree.agents.dqn import DQNAgent
from aitraineree.loggers.file_logger import FileLogger
from aitraineree.runners.env_runner import EnvRunner
from aitraineree.tasks import GymTask

seed = 32167
# torch.set_deterministic(True)
torch.manual_seed(seed)
data_logger = FileLogger(filepath="cart_dqn")

env_name = "CartPole-v1"
task = GymTask(env_name, seed=seed)
agent = DQNAgent(task.obs_space, task.action_space, n_steps=5, seed=seed)
env_runner = EnvRunner(task, agent, data_logger=data_logger, seed=seed)

scores = env_runner.run(reward_goal=100, max_episodes=300, force_new=True)
env_runner.interact_episode(render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.savefig(f"{env_name}.png", dpi=120)
plt.show()
