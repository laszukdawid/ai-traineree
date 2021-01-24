import pylab as plt

from ai_traineree.agents.ppo import PPOAgent as Agent
from ai_traineree.env_runner import EnvRunner
from ai_traineree.loggers import TensorboardLogger
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType


env_name = 'BipedalWalker-v3'
data_logger = TensorboardLogger()
task: TaskType = GymTask(env_name)
config = {
    'device': 'cuda',
    'num_epochs': 10,
    'rollout_length': 2048,
    'batch_size': 64,
    'simple_policy': True,
    'actor_number_updates': 1,
    'critic_number_updates': 1,

    "gae_lambda": 0.95,
    "ppo_ratio_clip": 0.20,
    "entropy_weight": 0.005,
    "gamma": 0.99,
    "std_init": 0.5,
    "std_max": 1.0,
    "std_min": 0.1,

    "max_grad_norm_actor": 200.0,
    "max_grad_norm_critic": 200.0,
    "critic_lr": 3e-4,
    "critic_betas": (0.9, 0.999),
    "actor_lr": 3e-4,
    "actor_betas": (0.9, 0.999),
}
agent = Agent(task.state_size, task.action_size, hidden_layers=(100, 100), **config)
env_runner = EnvRunner(task, agent, max_iterations=2000, data_logger=data_logger)
# env_runner.interact_episode(render=True)
scores = env_runner.run(300, 1000, log_episode_freq=1, gif_every_episodes=500, force_new=True)
env_runner.interact_episode(render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
