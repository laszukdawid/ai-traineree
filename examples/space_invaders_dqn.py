from collections import deque

import numpy as np
import pylab as plt
import torch
import torch.nn as nn

from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.loggers import TensorboardLogger
from ai_traineree.networks.bodies import ConvNet, FcNet, ScaleNet
from ai_traineree.networks.heads import NetChainer
from ai_traineree.runners.env_runner import EnvRunner
from ai_traineree.tasks import GymTask

# TODO: This needs internal handling. It's a common way to handle pixels as input, i.e. stack frames.
#       In the form right here this is a nasty and ugly hack.
prev_states = 2
states = deque(np.array([]), maxlen=prev_states)


def state_transform(img):
    state = np.mean(img, axis=2)[None, ...]
    states.append(state)
    state = np.vstack(states)
    return torch.from_numpy(state).float()


def network_fn(state_dim, output_dim, device):
    conv_net = ConvNet(state_dim, hidden_layers=(10, 10), device=device)
    return NetChainer(
        net_classes=[
            ScaleNet(scale=1.0 / 255),
            conv_net,
            nn.Flatten(),
            FcNet(conv_net.output_size, output_dim, hidden_layers=(100, 100, 50), device=device),
        ]
    )


env_name = "SpaceInvaders-v0"
data_logger = TensorboardLogger()
task = GymTask(env_name, state_transform=state_transform)
config = {
    "network_fn": lambda: network_fn(task.actual_obs_size, task.action_size, "cuda"),
    "compress_state": True,
    "gamma": 0.99,
    "lr": 1e-3,
    "update_freq": 150,
    "batch_size": 400,
    "buffer_size": int(5e3),
    "device": "cuda",
}

for _ in range(prev_states):
    task.reset()

agent = DQNAgent(task.obs_space, task.action_space, **config)
env_runner = EnvRunner(task, agent, data_logger=data_logger)

# env_runner.interact_episode(eps=0, render=True)
scores = env_runner.run(
    reward_goal=1000,
    max_episodes=20000,
    eps_start=0.9,
    gif_every_episodes=200,
    force_new=True,
)
# env_runner.interact_episode(render=True)
data_logger.close()

# plot scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.savefig(f"{env_name}.png", dpi=120)
plt.show()
