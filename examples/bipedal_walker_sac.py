import numpy as np
import pylab as plt

from ai_traineree.agents.sac import SACAgent as Agent
from ai_traineree.loggers import TensorboardLogger
from ai_traineree.runners.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType

data_logger = TensorboardLogger()
env_name = "BipedalWalker-v3"
task: TaskType = GymTask(env_name)
config = {
    "warm_up": 2000,
    "batch_size": 100,
    "update_freq": 10,
    "number_updates": 10,
    "actor_number_updates": 2,
    "critic_number_updates": 2,
    "gamma": 0.999,
    "critic_lr": 3e-4,
    "actor_lr": 3e-4,
    "alpha_lr": 3e-4,
    "alpha": 0.3,
    "tau": 0.005,
    "max_grad_norm_alpha": 1.0,
    "max_grad_norm_actor": 10.0,
    "max_grad_norm_critic": 10.0,
}
agent = Agent(task.obs_space, task.action_space, hidden_layers=(200, 200), **config)

env_runner = EnvRunner(task, agent, max_iterations=10000, data_logger=data_logger)
scores = env_runner.run(reward_goal=100, max_episodes=3000, eps_decay=0.99, log_episode_freq=1, force_new=True)
env_runner.interact_episode(render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.savefig(f"{env_name}.png", dpi=120)
plt.show()
