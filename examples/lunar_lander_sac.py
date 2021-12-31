# *Note* that this example isn't optimial but it should work.
import datetime

import pylab as plt
import torch

from ai_traineree.agents.sac import SACAgent
from ai_traineree.loggers import TensorboardLogger
from ai_traineree.runners.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType

seed = 321671
torch.manual_seed(seed)


env_name = "LunarLanderContinuous-v2"
task: TaskType = GymTask(env_name, seed=seed)
config = {
    "warm_up": 5000,
    "device": "cpu",
    "batch_size": 200,
    "update_freq": 500,
    "number_updates": 10,
    "actor_number_updates": 5,
    "critic_number_updates": 5,
    "max_grad_norm_actor": 10,
    "max_grad_norm_critic": 10,
    "max_grad_norm_alpha": 3,
    "hidden_layers": (200, 200),
    "actor_lr": 1e-3,
    "critic_lr": 3e-4,
    "alpha_lr": 3e-4,
    "tau": 0.005,
    "alpha": 0.3,
    "seed": seed,
    "simple_policy": True,
}
agent = SACAgent(task.obs_space, task.action_space, **config)

log_dir = f"runs/{env_name}_{agent.model}-{datetime.datetime.now().isoformat()[:-7]}"
data_logger = TensorboardLogger(log_dir=log_dir)
env_runner = EnvRunner(task, agent, data_logger=data_logger, seed=seed, debug_log=True)
scores = env_runner.run(
    reward_goal=30, max_episodes=2000, eps_end=0.01, eps_decay=0.95, force_new=True, checkpoint_every=None
)
env_runner.interact_episode(eps=0, render=True)
data_logger.close()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.savefig(f"{env_name}.png", dpi=120)
