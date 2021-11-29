"""
The example uses, but doesn't provide, a simplified version of Unity's Tennis environment [1].
Yes, this isn't cool, but just bare with me for until better examples are created.
In case you want recreate the environment yourself... go ahead. In short, open Tennis environment,
remove everything except the first pair and build. Below `build_path` should contain the output path.
Feel free to ping me but hopefully there are better examples before anyone sees this.

[1] https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md
"""
import pylab as plt
from mlagents_envs.environment import UnityEnvironment

from ai_traineree.multi_agent.maddpg import MADDPGAgent
from ai_traineree.runners.multiagent_env_runner import MultiAgentEnvRunner
from ai_traineree.tasks import MultiAgentUnityTask

build_path = "/home/kretyn/projects/ml-agents/Project/Builds/TennisSingle/tennis.x86_64"
unity_env = UnityEnvironment(build_path, no_graphics=True)

ma_task = MultiAgentUnityTask(unity_env=unity_env, allow_multiple_obs=True)
ma_task.reset()

obs_space = ma_task.observation_space[0].shape[0]
action_size = ma_task.action_space.shape[0]
num_agents = ma_task.num_agents
config = {
    "device": "cpu",
    "warm_up": 0,
    "update_freq": 10,
    "batch_size": 200,
}
ma_agent = MADDPGAgent(obs_space, action_size, num_agents, **config)


env_runner = MultiAgentEnvRunner(ma_task, ma_agent, max_iterations=200)
scores = env_runner.run(reward_goal=5, max_episodes=50, force_new=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.savefig("tennis.png", dpi=120)
plt.show()
