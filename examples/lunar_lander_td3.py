from ai_traineree.agents.td3 import TD3Agent as Agent
from ai_traineree.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType

import pylab as plt


env_name = 'LunarLanderContinuous-v2'
task: TaskType = GymTask(env_name)
config = {
    'action_scale': 2,
    'batch_size': 200,
    'number_updates': 5,
    'update_freq': 10,
    'update_policy_freq': 10,
}
agent = Agent(task.obs_size, task.action_size, **config)
env_runner = EnvRunner(task, agent)

# interact_episode(task, agent, 0, render=True)
scores = env_runner.run(reward_goal=80, max_episodes=2000, log_episode_freq=1, force_new=True)
env_runner.interact_episode(0, render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
