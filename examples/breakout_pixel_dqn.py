from ai_traineree.agents.dqn_pixels import DQNPixelAgent
from ai_traineree.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pylab as plt


def state_transform(state):
    """
    Simple cropping of the top and bottom edge and converting to blackwhite scale.
    """
    return state[30:-10].sum(-1) > 0


writer = SummaryWriter()

env_name = 'Breakout-v0'
task = GymTask(env_name, state_transform=state_transform)
state_size = np.array(task.reset()).shape

config = {"update_freq": 12, "batch_size": 40, "warm_up": 1000}
agent = DQNPixelAgent(state_size, task.action_size, hidden_layers=(200, 200))
env_runner = EnvRunner(task, agent, max_iterations=20000, writer=writer)

scores = env_runner.run(reward_goal=500, max_episodes=500, log_every=1, eps_start=0.8, gif_every_episodes=50)
env_runner.interact_episode(render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
