"""
This example uses DummyAgent on CartPole gym environment.
Since Dummy Agent only returns random values and never learns
this example is purely for demonsitration purpose only.
"""
import numpy as np
import pylab as plt
import torch

from ai_traineree.agents.dummy import DummyAgent
from ai_traineree.loggers import TensorboardLogger
from ai_traineree.runners.env_runner import EnvRunner
from ai_traineree.tasks import GymTask

seed = 32167
# torch.set_deterministic(True)
torch.manual_seed(seed)
data_logger = TensorboardLogger()

env_name = "CartPole-v1"
task = GymTask(env_name, seed=seed)
agent = DummyAgent(task.obs_space, task.action_space)
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
