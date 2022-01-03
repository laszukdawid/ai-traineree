import numpy as np
import pylab as plt

from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.runners.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType

env_name = "LunarLander-v2"
task: TaskType = GymTask(env_name)
config = {"batch_size": 64}
agent = DQNAgent(task.obs_space, task.action_space, config=config)
env_runner = EnvRunner(task, agent)

env_runner.interact_episode(eps=0, render=True)
scores = env_runner.run(50, 800, eps_start=1.0, eps_end=0.05, eps_decay=0.995)
env_runner.interact_episode(eps=0, render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.savefig(f"{env_name}.png", dpi=120)
plt.show()
