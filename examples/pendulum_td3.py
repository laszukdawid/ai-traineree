import numpy as np
import pylab as plt

from ai_traineree.agents.td3 import TD3Agent as Agent
from ai_traineree.runners.env_runner import EnvRunner
from ai_traineree.tasks import GymTask

env_name = "Pendulum-v0"
task = GymTask(env_name)
config = {
    "warm_up": 100,
    "batch_size": 50,
    "hidden_layers": (50, 50),
    "noise_scale": 1.0,
    "clip": (-2, 2),
    "actor_lr": 1e-4,
    "critic_lr": 2e-4,
}
agent = Agent(task.obs_space, task.action_space, **config)
env_runner = EnvRunner(task, agent)

# env_runner.interact_episode(eps=0, render=True)
scores = env_runner.run(0, 2000, eps_start=1.0, eps_end=0.05, eps_decay=0.99, log_episode_freq=1)
env_runner.interact_episode(eps=0, render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.savefig(f"{env_name}.png", dpi=120)
plt.show()
