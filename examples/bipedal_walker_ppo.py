from ai_traineree.agents.ppo import PPOAgent as Agent
from ai_traineree.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType

import numpy as np
import pylab as plt


env_name = 'BipedalWalker-v3'
task: TaskType = GymTask(env_name)
config = {
    'rollout_length': 100,
    'batch_size': 100,
    "number_updates": 1,
    "ppo_ratio_clip": 0.2,
    "value_loss_weight": 2,
    "entropy_weight": 0.0005,
    "gamma": 0.99,
    "action_scale": 2,
    "max_grad_norm_actor": 2.0,
    "max_grad_norm_critic": 2.0,
    "critic_lr": 2e-3,
    "actor_lr": 1e-3,
}
agent = Agent(task.state_size, task.action_size, hidden_layers=(500, 300), config=config)

env_runner = EnvRunner(task, agent, max_iterations=1000)
env_runner.interact_episode(render=True)
scores = env_runner.run(task, agent, 300, 200)
env_runner.interact_episode(render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
