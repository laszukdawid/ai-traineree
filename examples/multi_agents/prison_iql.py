import pylab as plt

from ai_traineree.multiagent_env_runner import MultiAgentCycleEnvRunner
from ai_traineree.multi_agents.iql import IQLAgent
from ai_traineree.tasks import PettingZooTask
from pettingzoo.butterfly import prison_v2 as prison

env = prison.env(vector_observation=True)
ma_task = PettingZooTask(env)
ma_task.reset()

state_size = ma_task.state_size
action_size = ma_task.action_size.n
agent_number = ma_task.num_agents
config = {
    'device': 'cpu',
    'warm_up': 0,
    'update_freq': 10,
    'batch_size': 200,
    'agent_names': env.agents,
}
ma_agent = IQLAgent(state_size, action_size, agent_number, **config)


env_runner = MultiAgentCycleEnvRunner(ma_task, ma_agent, max_iterations=200)
scores = env_runner.run(reward_goal=5, max_episodes=500, log_episode_freq=1, force_new=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('tennis.png', dpi=120)
plt.show()