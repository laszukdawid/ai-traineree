from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.env_runner import EnvRunner
from ai_traineree.tasks import GymTask

import pylab as plt
import gym


env_name = 'Breakout-ram-v0'
env = gym.make(env_name)

task = GymTask(env, env_name)
agent = DQNAgent(task.state_size, task.action_size, hidden_layers=(400, 300))
env_runner = EnvRunner(task, agent)

# env_runner.interact_episode(0, render=True)
scores = env_runner.run(reward_goal=5, max_episodes=5, print_every=1)
env_runner.interact_episode(100, render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
