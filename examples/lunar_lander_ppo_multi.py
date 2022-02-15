from typing import List

import pylab as plt

from ai_traineree.agents.ppo import PPOAgent as Agent
from ai_traineree.loggers.tensorboard_logger import TensorboardLogger
from ai_traineree.runners.multi_sync_env_runner import MultiSyncEnvRunner
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType

env_name = "LunarLanderContinuous-v2"
data_logger = TensorboardLogger()
processes = 4
num_workers = processes
kwargs = {
    # "device": "cuda",
    "num_workers": num_workers,
    "rollout_length": 200,
    "batch_size": 200,
    "actor_number_updates": 5,
    "critic_number_updates": 5,
    "using_gae": True,  # Default is True
    "hidden_layers": (64, 64),
    "ppo_ratio_clip": 0.2,
    "entropy_weight": 0.05,
    "max_grad_norm_actor": 5.0,
    "max_grad_norm_critic": 5.0,
    "critic_lr": 1e-4,
    "actor_lr": 3e-4,
    "using_kl_div": False,
    "simple_policy": True,
}
tasks: List[TaskType] = [GymTask(env_name) for _ in range(num_workers)]
agent = Agent(tasks[0].obs_space, tasks[0].action_space, **kwargs)
env_runner = MultiSyncEnvRunner(tasks, agent, processes=processes, data_logger=data_logger)
scores = env_runner.run(reward_goal=80, eps_end=0.01, max_episodes=4000, force_new=True, log_episode_freq=1)

data_logger.close()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.savefig(f"{env_name}.png", dpi=120)
plt.show()
