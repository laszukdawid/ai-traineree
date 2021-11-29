import pylab as plt

from ai_traineree.agents.ppo import PPOAgent as Agent
from ai_traineree.loggers import TensorboardLogger
from ai_traineree.runners.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType

env_name = "BipedalWalker-v3"
data_logger = TensorboardLogger()
task: TaskType = GymTask(env_name)
config = {
    "device": "cuda",
    "num_epochs": 1,
    "rollout_length": 2000,
    "batch_size": 2000,
    "simple_policy": True,
    "number_updates": 40,
    "hidden_layers": (100, 100),
    "gae_lambda": 0.95,
    "ppo_ratio_clip": 0.20,
    "entropy_weight": 0.005,
    "gamma": 0.99,
    "std_min": 0,
    "std_max": 10,
    "std_init": 0.6,
    "max_grad_norm_actor": 200.0,
    "max_grad_norm_critic": 200.0,
    "critic_lr": 1e-3,
    "actor_lr": 3e-4,
}
agent = Agent(task.obs_space, task.action_space, **config)
env_runner = EnvRunner(task, agent, max_iterations=2000, data_logger=data_logger)
# env_runner.interact_episode(render=True)
scores = env_runner.run(300, 2000, eps_end=0.001, eps_decay=0.9, log_episode_freq=10, force_new=True)
env_runner.interact_episode(render=True)

data_logger.close()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.savefig(f"{env_name}.png", dpi=120)
plt.show()
