import pylab as plt

from ai_traineree.agents.ppo import PPOAgent as Agent
from ai_traineree.loggers.file_logger import FileLogger
from ai_traineree.runners.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType

env_name = "LunarLanderContinuous-v2"
task: TaskType = GymTask(env_name)
config = {
    "rollout_length": 500,
    "batch_size": 500,
    "number_updates": 60,
    "hidden_layers": (64, 64),
    "using_gae": True,  # Default is True
    "ppo_ratio_clip": 0.1,
    "entropy_weight": 0.05,
    "max_grad_norm_actor": 1.0,
    "max_grad_norm_critic": 1.0,
    "critic_lr": 1e-3,
    "actor_lr": 3e-3,
    "using_kl_div": True,
    "simple_policy": True,
}
agent = Agent(task.obs_space, task.action_space, **config)
data_logger = FileLogger(f"{agent.model}_{env_name}")

env_runner = EnvRunner(task, agent, data_logger=data_logger)
scores = env_runner.run(
    reward_goal=100, max_episodes=2000, eps_decay=0.9, eps_end=0.001, force_new=True, log_episode_freq=10
)
env_runner.interact_episode(eps=0, render=True)

data_logger.close()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.savefig(f"{agent.model}_{env_name}.png", dpi=120)
plt.show()
