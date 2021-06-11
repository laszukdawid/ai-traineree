from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.env_runner import EnvRunner
from ai_traineree.tasks import GymTask

import pylab as plt


env_name = 'Breakout-ram-v0'
config = {
    "hidden_layers": (500, 400),
    "lr": 1e-3,
    "device": "cuda",
}
task = GymTask(env_name)
agent = DQNAgent(task.obs_size, task.action_size, **config)
env_runner = EnvRunner(task, agent)

# env_runner.interact_episode(0, render=True)
scores = env_runner.run(reward_goal=5, max_episodes=5000, log_every=1)
env_runner.interact_episode(100, render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
