from ai_traineree.agents.ddpg import DDPGAgent as DDPG
from ai_traineree.env_runner import EnvRunner
from ai_traineree.loggers import TensorboardLogger
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType

import pylab as plt


data_logger = TensorboardLogger()
env_name = 'LunarLanderContinuous-v2'
task: TaskType = GymTask(env_name)
config = {'action_scale': 1, 'update_freq': 2}
agent = DDPG(task.obs_space, task.action_space, hidden_layers=(100, 100), noise_scale=0.4, noise_sigma=0.2, **config)
env_runner = EnvRunner(task, agent, data_logger=data_logger)
scores = env_runner.run(reward_goal=80, max_episodes=1000, eps_start=1.0, eps_end=0.05, eps_decay=0.999, force_new=True)
# env_runner.interact_episode(0, render=True)
data_logger.close()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
