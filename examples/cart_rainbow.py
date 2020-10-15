import pylab as plt
import numpy as np
import sneks  # noqa

from ai_traineree.agents.rainbow import RainbowAgent as Agent
# from ai_traineree.agents.dqn import DQNAgent as Agent
from ai_traineree.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from torch.utils.tensorboard import SummaryWriter


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


writer = SummaryWriter()
# env_name = 'babysnek-raw-16-v1'
env_name = 'CartPole-v1'
task = GymTask(env_name)
state_size = np.array(task.reset()).shape

config = {
    "warm_up": 200,
    "update_freq": 10,
    "number_updates": 1,
    "batch_size": 50,
    "lr": 2.5e-3,
    "n_steps": 1,
    "tau": 0.001,
    "max_grad_norm": 5.0,  # 5
    "hidden_layers": (500, 500),
    "v_min": 0,
    "v_max": 20,
    "n_atoms": 11,
    # "device": "cpu",
}

agent = Agent(state_size, task.action_size, **config)
env_runner = EnvRunner(task, agent, max_iterations=2000, writer=writer)

scores = env_runner.run(
    # reward_goal=0.75,
    reward_goal=100,
    max_episodes=10000,
    log_every=5,
    eps_start=0.99, eps_decay=0.99, eps_end=0.01,
    force_new=True
)
env_runner.interact_episode(render=True)


avg_length = 100
ma = running_mean(scores, avg_length)
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.plot(range(avg_length, avg_length+len(ma)), ma)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
