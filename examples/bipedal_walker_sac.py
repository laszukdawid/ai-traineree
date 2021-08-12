from ai_traineree.agents.sac import SACAgent as Agent
from ai_traineree.env_runner import EnvRunner
from ai_traineree.loggers import TensorboardLogger
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType

import numpy as np
import pylab as plt


data_logger = TensorboardLogger()
env_name = 'BipedalWalker-v3'
task: TaskType = GymTask(env_name)
config = {
    'warm_up': 500,
    'batch_size': 200,
    'update_freq': 30,
    "number_updates": 1,
    "gamma": 0.99,
    "critic_lr": 1e-3,
    "actor_lr": 2e-3,
    "alpha": 0.2,
    "tau": 0.01,
    "max_grad_norm_alpha": 1.0,
    "max_grad_norm_actor": 10.0,
    "max_grad_norm_critic": 10.0,
}
agent = Agent(task.obs_space, task.action_space, hidden_layers=(100, 100), **config)

env_runner = EnvRunner(task, agent, max_iterations=10000, data_logger=data_logger)
# env_runner.interact_episode(render=True)
scores = env_runner.run(reward_goal=10, max_episodes=500, eps_decay=0.99, log_episode_freq=1, gif_every_episodes=200, force_new=True)
env_runner.interact_episode(render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
