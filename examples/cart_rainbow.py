import numpy as np
import pylab as plt

from ai_traineree.agents.rainbow import RainbowAgent as Agent
from ai_traineree.loggers.tensorboard_logger import TensorboardLogger
from ai_traineree.runners.env_runner import EnvRunner
from ai_traineree.tasks import GymTask


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


env_name = "CartPole-v1"
task = GymTask(env_name)
data_logger = TensorboardLogger()

agent = Agent(task.obs_space, task.action_space, device="cpu")
env_runner = EnvRunner(task, agent, data_logger=data_logger)

scores = env_runner.run(reward_goal=100, max_episodes=500, eps_decay=0.98, force_new=True)
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
