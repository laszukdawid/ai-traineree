# **NOTE** from @dawid 2021-06-21:
#   This example (likely) doesn't work.
#   PPO has been tested and proven to work on environments with a single agent
#   so the suspicion is that there is a bug in MultiEnvRunner or PPO processing multi agents.


import os
from typing import List

import pylab as plt

from ai_traineree.agents.ppo import PPOAgent as Agent

# from ai_traineree.loggers import NeptuneLogger
from ai_traineree.loggers import TensorboardLogger
from ai_traineree.runners.env_runner import MultiSyncEnvRunner
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType

neptune_api_key = os.getenv("NEPTUNE_API_TOKEN")

env_name = "LunarLanderContinuous-v2"
# data_logger = NeptuneLogger("kretyn/PPO-LunarLander-Multi", api_token=neptune_api_key)
# data_logger= None
data_logger = TensorboardLogger()
processes = 1
num_workers = processes
kwargs = {
    "device": "cuda",
    "num_workers": num_workers,
    "rollout_length": 1000,
    "batch_size": 1000,
    "number_updates": 80,
    "using_gae": True,  # Default is True
    "hidden_layers": (64, 64),
    "ppo_ratio_clip": 0.1,
    "entropy_weight": 0.005,
    "critic_lr": 1e-3,
    "actor_lr": 3e-4,
    "max_grad_norm_actor": 20.0,
    "max_grad_norm_critic": 20.0,
    "target_kl": 0.02,
}
tasks: List[TaskType] = [GymTask(env_name) for _ in range(num_workers)]
agent = Agent(tasks[0].obs_space, tasks[0].action_space, **kwargs)
env_runner = MultiSyncEnvRunner(tasks, agent, processes=processes, data_logger=data_logger)
scores = env_runner.run(reward_goal=80, eps_end=0.01, max_episodes=5000, force_new=True, log_episode_freq=10)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.savefig(f"{env_name}.png", dpi=120)
plt.show()
