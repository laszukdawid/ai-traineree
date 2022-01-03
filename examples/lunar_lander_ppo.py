import pylab as plt

from ai_traineree.agents.ppo import PPOAgent as Agent
from ai_traineree.loggers import TensorboardLogger
from ai_traineree.runners.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType

env_name = "LunarLanderContinuous-v2"
data_logger = TensorboardLogger()
task: TaskType = GymTask(env_name)
config = {
    "rollout_length": 500,
    "batch_size": 500,
    "number_updates": 60,
    "hidden_layers": (64, 64),
    "using_gae": True,  # Default is True
    "ppo_ratio_clip": 0.1,
    "entropy_weight": 0.05,
    "action_scale": 1,
    "max_grad_norm_actor": 10.0,
    "max_grad_norm_critic": 10.0,
    "critic_lr": 1e-4,
    "actor_lr": 3e-4,
    "using_kl_div": True,
    "std_min": 0,
    "std_max": 10,
    "std_init": 0.6,
    "simple_policy": True,
}
agent = Agent(task.obs_space, task.action_space, **config)
env_runner = EnvRunner(task, agent, data_logger=data_logger)
# env_runner.interact_episode(eps=0, render=True)
scores = env_runner.run(100, 2000, eps_decay=0.9, eps_end=0.001, force_new=True, log_episode_freq=10)
env_runner.interact_episode(eps=0, render=True)

data_logger.close()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.savefig(f"{env_name}.png", dpi=120)
plt.show()
