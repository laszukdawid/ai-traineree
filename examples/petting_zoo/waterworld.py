"""
This example is for demonstartion purpose only.
No agent learns here anything useful, yet.

Well, maybe they do but it might take a long time to check.
Ain't nobody got time for that.
"""
from pettingzoo.sisl import waterworld_v3

from ai_traineree.agents.ppo import PPOAgent
from ai_traineree.multi_agents.independent import IndependentAgents
from ai_traineree.runners.multiagent_env_runner import MultiAgentCycleEnvRunner
from ai_traineree.tasks import PettingZooTask

env = waterworld_v3.env()
task = PettingZooTask(env=env)
task.reset()  # Needs to be reset to access env.agents()

agents = []
for actor_name in task.agents:
    obs_space = task.observation_spaces[actor_name]
    action_space = task.action_spaces[actor_name]

    agents.append(PPOAgent(obs_space, action_space))


multi_agent = IndependentAgents(agents, agent_names=task.agents)
runner = MultiAgentCycleEnvRunner(task, multi_agent=multi_agent)
runner.run(max_episodes=3)
