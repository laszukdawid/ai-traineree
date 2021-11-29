import datetime

import pylab as plt
import torch.nn as nn

from ai_traineree.agents.rainbow import RainbowAgent
from ai_traineree.loggers import TensorboardLogger
from ai_traineree.networks.bodies import ConvNet, FcNet
from ai_traineree.networks.heads import NetChainer
from ai_traineree.runners.env_runner import EnvRunner
from ai_traineree.tasks import GymTask


def state_transform(state):
    """
    Simple cropping of the top and bottom edge and converting to blackwhite scale.
    """
    return state.mean(2)[None, ...] / 256.0


def agent_state_tranform(state):
    return state


def network_fn(state_dim, output_dim, device=None):
    conv_net = ConvNet(
        state_dim, hidden_layers=(30, 30), kernel_sze=(16, 8), max_pool_size=(4, 2), stride=(4, 2), device=device
    )
    return NetChainer(
        net_classes=[
            conv_net,
            nn.Flatten(),
            FcNet(conv_net.output_size, output_dim, hidden_layers=(200, 200), device=device),
        ]
    )


env_name = "SpaceInvaders-v0"
task = GymTask(env_name, state_transform=state_transform)

device = "cuda"
config = {
    "device": device,
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
agent = RainbowAgent(task.obs_space, task.action_space, **config)
data_logger = TensorboardLogger(f'runs/{env_name}_{agent.model}_{datetime.datetime.now().strftime("%b%d_%H-%m-%s")}')
env_runner = EnvRunner(task, agent, max_iterations=10000, data_logger=data_logger)

scores = env_runner.run(reward_goal=1000, max_episodes=1000, eps_start=0.99, gif_every_episodes=100, force_new=True)
env_runner.interact_episode(render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.savefig(f"{env_name}.png", dpi=120)
plt.show()
