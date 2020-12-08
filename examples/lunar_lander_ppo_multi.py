import pylab as plt

from ai_traineree.agents.ppo import PPOAgent as Agent
from ai_traineree.env_runner import MultiSyncEnvRunner
from ai_traineree.tasks import GymTask
from ai_traineree.types import TaskType
from torch.utils.tensorboard import SummaryWriter


env_name = 'LunarLanderContinuous-v2'
writer = SummaryWriter()
task: TaskType = GymTask(env_name)
processes = 12
executor_num = processes
kwargs = {
    'executor_num': executor_num,
    'device': 'cpu',
    'rollout_length': 40,
    'batch_size': 40 * processes,
    "number_updates": 1,

    "value_loss_weight": 1e-1,
    "entropy_weight": 2e-2,
    "gamma": 0.99,
    "critic_lr": 1e-3,
    "actor_lr": 1e-4,
}
agent = Agent(task.state_size, task.action_size, hidden_layers=(200, ), **kwargs)
tasks = [GymTask(env_name) for _ in range(executor_num)]
env_runner = MultiSyncEnvRunner(tasks, agent, processes=processes, writer=writer)
scores = env_runner.run(reward_goal=80, max_episodes=10000, force_new=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
