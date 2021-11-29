import numpy as np
import pylab as plt
import sneks  # noqa

from ai_traineree.agents.rainbow import RainbowAgent
from ai_traineree.loggers import TensorboardLogger
from ai_traineree.runners.env_runner import EnvRunner
from ai_traineree.tasks import GymTask


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def state_transform(state: np.ndarray):
    return state[1:-1, 1:-1].flatten()


def reward_transform(reward) -> float:
    if reward == 0:
        return -0.005
    else:
        return reward


data_logger = TensorboardLogger()
env_name = "hungrysnek-raw-16-v1"
task = GymTask(env_name, state_transform=state_transform, reward_transform=reward_transform)
obs_size = np.array(task.reset()).shape

device = "cuda"
config = {
    "warm_up": 500,
    "update_freq": 10,
    "number_updates": 2,
    "batch_size": 100,
    "lr": 2e-4,
    "n_steps": 3,
    "tau": 0.01,
    "max_grad_norm": 10.0,
    "hidden_layers": (1200, 1000),
}


agent = RainbowAgent(task.obs_space, task.action_space, **config)
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
