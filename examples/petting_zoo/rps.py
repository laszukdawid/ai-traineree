from pettingzoo.classic import rps_v1

from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.tasks import PettingZooTask

env = rps_v1.env()
task = PettingZooTask(env=env)
task.reset()  # Needs to be reset to access env.agents()


agents = {}
for actor_name in task.agents:
    obs_space = task.observation_spaces[actor_name]
    action_space = task.action_spaces[actor_name]

    agents[actor_name] = DQNAgent(obs_space, action_space)

task.reset()

for actor_name in task.agent_iter(max_iter=10000):
    last = task.last()
    obs, reward, done, info = last

    agent = agents[actor_name]
    action = agent.act(obs[None, ...])

    if done:
        print(f"Finished. Actor: {actor_name};\tReward: {reward}")
        break

    task.step(action)
