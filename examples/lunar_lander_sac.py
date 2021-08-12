import datetime
import pylab as plt
import torch

from ai_traineree.agents.sac import SACAgent as Agent
from ai_traineree.env_runner import EnvRunner
from ai_traineree.loggers import TensorboardLogger
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType


def reward_transform(reward):
    """Cap reward to [-1, 1]"""
    return max(-1, min(reward, 1))


seed = 32167
torch.manual_seed(seed)


env_name = 'LunarLanderContinuous-v2'
task: TaskType = GymTask(env_name, seed=seed, reward_transform=reward_transform)
config = {
    'warm_up': 400,
    'device': 'cpu',
    'batch_size': 60,
    'update_freq': 2,
    'number_updates': 1,
    'hidden_layers': (100, 100),
    'actor_lr': 5e-4,
    'critic_lr': 5e-4,
    'alpha_lr': 3e-5,
    'tau': 0.02,
    "alpha": 0.2,

    'seed': seed,
}
agent = Agent(task.obs_size, task.action_size, **config)

log_dir = f"runs/{env_name}_{agent.name}-{datetime.datetime.now().isoformat()[:-7]}"
data_logger = TensorboardLogger(log_dir=log_dir)
env_runner = EnvRunner(task, agent, data_logger=data_logger, seed=seed)
scores = env_runner.run(reward_goal=30, max_episodes=500, eps_end=0.01, eps_decay=0.95, force_new=True)
env_runner.interact_episode(0, render=True)
data_logger.close()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
