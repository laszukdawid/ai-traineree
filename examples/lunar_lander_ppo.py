import pylab as plt

from ai_traineree.agents.ppo import PPOAgent as Agent
from ai_traineree.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType
from torch.utils.tensorboard import SummaryWriter


env_name = 'LunarLanderContinuous-v2'
writer = SummaryWriter()
task: TaskType = GymTask(env_name)
config = {
    'rollout_length': 60,
    'batch_size': 60,
    "number_updates": 1,

    "using_gae": False,  # Default is True
    "ppo_ratio_clip": 0.2,
    "entropy_weight": 0.0005,
    "gamma": 0.99,
    "action_scale": 1,
    "max_grad_norm_actor": 3.0,
    "max_grad_norm_critic": 5.0,
    "critic_lr": 0.001,
    "actor_lr": 0.0004,
}
agent = Agent(task.state_size, task.action_size, hidden_layers=(80, 80), **config)
env_runner = EnvRunner(task, agent, writer=writer)
# env_runner.interact_episode(0, render=True)
scores = env_runner.run(80, 8000, force_new=True, eps_decay=0.99)
env_runner.interact_episode(0, render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
