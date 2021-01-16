import pylab as plt
import os

from ai_traineree.agents.ppo import PPOAgent as Agent
from ai_traineree.env_runner import MultiSyncEnvRunner
from ai_traineree.loggers import NeptuneLogger
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType
from typing import List


neptune_api_key = os.getenv("NEPTUNE_API_TOKEN")

env_name = 'LunarLanderContinuous-v2'
data_logger = NeptuneLogger("kretyn/PPO-LunarLander-Multi", api_token=neptune_api_key)
processes = 4
num_workers = processes
kwargs = {
    'device': "cuda",
    'num_workers': num_workers,
    'num_epochs': 4,
    'rollout_length': 128,
    'batch_size': 64,
    "actor_number_updates": 4,
    "critic_number_updates": 4,
    "std_init": 0.8,
    "std_max": 1.1,
    "std_min": 0.05,
    "ppo_ratio_clip": 0.2,
    "simple_policy": True,

    "using_kl_div": True,
    # "value_loss_weight": 2,
    "entropy_weight": 0.01,
    "gamma": 0.999,
    'lambda_gae': 0.98,
    "critic_lr": 3e-4,
    "actor_lr": 3e-4,
    "action_scale": 1,
    "action_min": -20,
    "action_max": 20,
}
tasks: List[TaskType] = [GymTask(env_name) for _ in range(num_workers)]
agent = Agent(tasks[0].state_size, tasks[0].action_size, hidden_layers=(100, 64, 64), **kwargs)
env_runner = MultiSyncEnvRunner(tasks, agent, processes=processes, data_logger=data_logger)
scores = env_runner.run(reward_goal=80, max_episodes=5000, force_new=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
