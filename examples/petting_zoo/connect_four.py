from pettingzoo.classic import connect_four_v3

from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.runners.multiagent_env_runner import MultiAgentCycleEnvRunner
from ai_traineree.tasks import PettingZooTask

env = connect_four_v3.env()
task = PettingZooTask(env=env)
task.reset()  # Needs to be reset to access env.agents()


agents = []
for actor_name in env.agent_iter():
    obs_space = task.observation_spaces[actor_name]
    action_space = task.action_spaces[actor_name]

    agent = DQNAgent(obs_space, action_space)
    agents.append(agent)


runner = MultiAgentCycleEnvRunner(task, agents, mode="compete")
