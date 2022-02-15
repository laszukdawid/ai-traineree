from collections import defaultdict

import pylab as plt
from pettingzoo.sisl import multiwalker_v7

from ai_traineree.loggers.tensorboard_logger import TensorboardLogger
from ai_traineree.multi_agent.maddpg import MADDPGAgent
from ai_traineree.runners.multiagent_env_runner import MultiAgentCycleEnvRunner
from ai_traineree.tasks import PettingZooTask

env = multiwalker_v7.env()
ma_task = PettingZooTask(env)
ma_task.reset()

obs_size = int(ma_task.obs_size[0])
action_size = int(ma_task.action_size.shape[0])
agent_number = ma_task.num_agents
config = {
    "device": "cuda",
    "update_freq": 20,
    "batch_size": 200,
    "agent_names": env.agents,
    "hidden_layers": (500, 300, 100),
}
ma_agent = MADDPGAgent(obs_size, action_size, agent_number, **config)
data_logger = TensorboardLogger(log_dir="runs/Multiwalkers-MADDPG")
# data_logger = None

env_runner = MultiAgentCycleEnvRunner(ma_task, ma_agent, max_iterations=9000, data_logger=data_logger)
scores = env_runner.run(reward_goal=20, max_episodes=50, eps_decay=0.99, log_episode_freq=1, force_new=True)

parsed_scores = defaultdict(list)
summed_score = []
for score in scores:
    summed_score.append(0)
    for name, value in score.items():
        parsed_scores[name].append(value)
        summed_score[-1] += value

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(211)
for label, values in parsed_scores.items():
    plt.plot(range(len(scores)), values, label=label)
plt.ylabel("Score")
plt.xlabel("Episode #")

ax = fig.add_subplot(212)
plt.plot(range(len(scores)), summed_score)
plt.ylabel("Summed score")
plt.xlabel("Episode #")

plt.savefig("prison.png", dpi=120)
plt.show()
