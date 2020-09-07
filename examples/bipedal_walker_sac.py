from ai_traineree.agents.sac import SACAgent as Agent
from ai_traineree.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pylab as plt


writer = SummaryWriter()
env_name = 'BipedalWalker-v3'
task: TaskType = GymTask(env_name)
config = {
    'batch_size': 100,
    "number_updates": 2,
    "gamma": 0.99,
    "critic_lr": 2e-4,
    "actor_lr": 1e-3,
}
agent = Agent(task.state_size, task.action_size, hidden_layers=(200, 200), writer=writer, **config)

env_runner = EnvRunner(task, agent, max_iterations=10000, writer=writer)
# env_runner.interact_episode(render=True)
scores = env_runner.run(reward_goal=100, max_episodes=500, log_every=1)
env_runner.interact_episode(render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
