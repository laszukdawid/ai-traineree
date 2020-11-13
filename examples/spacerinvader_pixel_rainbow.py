from ai_traineree.agents.rainbow import RainbowAgent
from ai_traineree.networks.bodies import ConvNet, FcNet
from ai_traineree.networks.heads import NetChainer, RainbowNet
from ai_traineree.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pylab as plt
import torch.nn as nn


def state_transform(state):
    """
    Simple cropping of the top and bottom edge and converting to blackwhite scale.
    """
    return state.mean(2)[None, ...]/256.


def agent_state_tranform(state):
    return state


def network_fn(state_dim, output_dim, device=None):
    conv_net = ConvNet(state_dim, hidden_layers=(30, 30), kernel_sze=(16, 8), max_pool_size=(4, 2), stride=(4, 2), device=device)
    return NetChainer(net_classes=[
        conv_net,
        nn.Flatten(),
        FcNet(conv_net.output_size, output_dim, hidden_layers=(200, 200), device=device),
    ])


env_name = 'SpaceInvaders-v0'
task = GymTask(env_name, state_transform=state_transform)
writer = SummaryWriter()

device = "cuda"
config = {
    'device': device,
    "update_freq": 50,
    "number_updates": 5,
    "batch_size": 200,
    "buffer_size": 1e4,
    "warm_up": 100,
    "lr": 1e-4,
    "pre_network_fn": lambda in_features: network_fn(in_features, 300, device),
    "hidden_layers": None,
    "state_transform": agent_state_tranform,
}
state_size = task.actual_state_size
agent = RainbowAgent(state_size, task.action_size, **config)
env_runner = EnvRunner(task, agent, max_iterations=2000, writer=writer)

scores = env_runner.run(reward_goal=1000, max_episodes=1000, log_every=1, eps_start=0.99, gif_every_episodes=100, force_new=True)
env_runner.interact_episode(render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
