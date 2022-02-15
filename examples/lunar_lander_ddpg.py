import pylab as plt

from ai_traineree.agents.ddpg import DDPGAgent as DDPG
from ai_traineree.loggers.tensorboard_logger import TensorboardLogger
from ai_traineree.runners.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType

seed = 123
data_logger = TensorboardLogger()
env_name = "LunarLanderContinuous-v2"
task: TaskType = GymTask(env_name)
task.seed(seed)

config = {
    "warm_up": 1000,
    "update_freq": 50,
    "hidden_layers": (100, 100),
    "noise_scale": 1.0,
    "noise_sigma": 1.0,
    "number_updates": 1,
    "buffer_size": 1e5,
    "batch_size": 200,
    "tau": 0.002,
    "max_grad_norm_actor": 2.0,
    "max_grad_norm_critic": 2.0,
}
agent = DDPG(task.obs_space, task.action_space, **config)
env_runner = EnvRunner(task, agent, data_logger=data_logger, debug_log=True)
scores = env_runner.run(
    reward_goal=80,
    max_episodes=3000,
    eps_start=1.0,
    eps_end=0.3,
    eps_decay=0.999,
    force_new=True,
    checkpoint_every=None,
)
# env_runner.interact_episode(eps=0, render=True)
data_logger.close()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.savefig(f"{env_name}.png", dpi=120)
plt.show()
