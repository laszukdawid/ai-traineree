from ai_traineree.agents.td3 import TD3Agent as Agent
from ai_traineree.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType

import pylab as plt


env_name = 'LunarLanderContinuous-v2'
task: TaskType = GymTask(env_name)
config = {
    'batch_size': 50,
    'warm_up': 100,
    'action_scale': 2,
    'update_freq': 10,
    'hidden_layers': (200, 200),
    'noise_scale': 1.0,
    'noise_sigma': 0.2,
    'actor_lr': 1e-4,
}
agent = Agent(task.state_size, task.action_size, **config)
env_runner = EnvRunner(task, agent)

# interact_episode(task, agent, 0, render=True)
scores = env_runner.run(reward_goal=80, max_episodes=1000, eps_start=1.0, eps_end=0.05, eps_decay=0.991, log_every=1)
env_runner.interact_episode(0, render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
