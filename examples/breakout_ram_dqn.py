import pylab as plt

from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.runners.env_runner import EnvRunner
from ai_traineree.tasks import GymTask

env_name = "Breakout-ram-v0"
config = {
    "hidden_layers": (500, 400),
    "lr": 1e-3,
    "device": "cuda",
}
task = GymTask(env_name)
agent = DQNAgent(task.obs_space, task.action_space, **config)
env_runner = EnvRunner(task, agent)

# env_runner.interact_episode(eps=0, render=True)
scores = env_runner.run(reward_goal=5, max_episodes=5000)
env_runner.interact_episode(eps=0, render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.savefig(f"{env_name}.png", dpi=120)
plt.show()
