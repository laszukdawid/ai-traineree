import numpy as np
import pylab as plt
import sneks  # noqa
import torch.nn as nn

from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.loggers import TensorboardLogger
from ai_traineree.networks.bodies import ConvNet, FcNet, ScaleNet
from ai_traineree.networks.heads import NetChainer
from ai_traineree.runners.env_runner import EnvRunner
from ai_traineree.tasks import GymTask


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def state_transform(state: np.ndarray):
    return state[1:-1, 1:-1][None, ...]


def network_fn(state_dim, output_dim, device):
    conv_net = ConvNet(state_dim, hidden_layers=(10, 10), device=device)
    return NetChainer(
        net_classes=[
            conv_net,
            nn.Flatten(),
            FcNet((conv_net.output_size,), (output_dim,), hidden_layers=(100, 50), device=device),
        ]
    )


data_logger = TensorboardLogger()
env_name = "hungrysnek-raw-16-v1"
task = GymTask(env_name, state_transform=state_transform)

input_obs_shape = (1,) + task.obs_space.shape  # Reformat to (channels, height, width)
config = {
    "device": "cuda",
    "warm_up": 500,
    "update_freq": 10,
    "number_updates": 2,
    "batch_size": 100,
    "lr": 2e-4,
    "n_steps": 3,
    "tau": 0.01,
    "max_grad_norm": 10.0,
    "hidden_layers": (1200, 1000),
    "network_fn": lambda: network_fn(input_obs_shape, task.action_size, "cuda"),
}


# obs_space = (1,) + task.obs_space.shape
agent = DQNAgent(task.obs_space, task.action_space, **config)
env_runner = EnvRunner(task, agent, max_iterations=2000, data_logger=data_logger)

scores = env_runner.run(reward_goal=0.75, max_episodes=50000, gif_every_episodes=1000, force_new=True)
env_runner.interact_episode(render=True)
data_logger.close()


avg_length = 100
ma = running_mean(scores, avg_length)
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.plot(range(avg_length, avg_length + len(ma)), ma)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.savefig(f"{env_name}.png", dpi=120)
plt.show()
