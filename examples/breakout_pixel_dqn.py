from ai_traineree.networks import QNetwork2D
from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pylab as plt


def state_transform(state):
    """
    Simple cropping of the top and bottom edge and converting to blackwhite scale.
    """
    return (state[40:-10].sum(-1) > 0)[None, ...]


def agent_state_tranform(state):
    return state


env_name = 'Breakout-v0'
task = GymTask(env_name, state_transform=state_transform)
state_size = np.array(task.reset()).shape
writer = SummaryWriter()

config = {
    "update_freq": 10,
    "batch_size": 100,
    "warm_up": 100,
    "lr": 1e-4,
    "network_fn": lambda: QNetwork2D(state_size, task.action_size, hidden_layers=(200, 200)),
    "state_transform": agent_state_tranform,
}
agent = DQNAgent(state_size, task.action_size, **config)
env_runner = EnvRunner(task, agent, max_iterations=2000, writer=writer)

scores = env_runner.run(reward_goal=500, max_episodes=1000, log_every=1, eps_start=0.99, gif_every_episodes=100)
env_runner.interact_episode(render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
