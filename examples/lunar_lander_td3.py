from datetime import datetime

import pylab as plt
import torch

from ai_traineree.agents.td3 import TD3Agent as Agent
from ai_traineree.loggers.tensorboard_logger import TensorboardLogger
from ai_traineree.runners.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType

seed = 32167
torch.manual_seed(seed)


env_name = "LunarLanderContinuous-v2"
task: TaskType = GymTask(env_name, seed=seed)
config = {
    "warm_up": 1000,
    "batch_size": 100,
    "hidden_layers": (100, 100),
    "number_updates": 3,
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "update_freq": 20,
    "update_policy_freq": 40,
    "noise_scale": 1.0,
}
agent = Agent(task.obs_space, task.action_space, **config)

log_dir = f"runs/{env_name}_{agent.model}-{datetime.now().isoformat()[:-7]}"
data_logger = TensorboardLogger(log_dir=log_dir)
env_runner = EnvRunner(task, agent, data_logger=data_logger, window_len=30, debug_log=True)

# interact_episode(task, agent, 0, render=True)
scores = env_runner.run(reward_goal=80, max_episodes=2000, eps_end=0.2, eps_decay=0.99, force_new=True)
# env_runner.interact_episode(eps=0, render=True)

data_logger.close()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.savefig(f"{env_name}.png", dpi=120)
plt.show()
