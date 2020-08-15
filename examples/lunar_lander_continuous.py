from ai_traineree.agents.ddpg import DDPGAgent as DDPG
from ai_traineree.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType

import pylab as plt
import gym


env_name = 'LunarLanderContinuous-v2'
env = gym.make(env_name)

task: TaskType = GymTask(env, env_name)
config = {'batch_size': 64, 'warm_up': 0, 'action_scale': 2, 'update_freq': 2}
agent = DDPG(task.state_size, task.action_size, hidden_layers=(300, 200), noise_scale=0.4, noise_sigma=0.2, config=config)
env_runner = EnvRunner(task, agent)

# interact_episode(task, agent, 0, render=True)
scores = env_runner.run(reward_goal=80, max_episodes=400, eps_start=1.0, eps_end=0.05, eps_decay=0.991)
env_runner.interact_episode(0, render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
