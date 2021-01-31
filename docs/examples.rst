Examples
========

Here are only some selected examples. There are many more examples provided in the repository as individual files.
There is `examples` directory or directly here https://github.com/laszukdawid/ai-traineree/tree/master/examples.

Single agent
------------

DQN on CartPole
```````````````

This example uses the `CartPole` environment provided by the `OpenAI Gym <https://gym.openai.com/>`_.
If you don't have the `Gym` then you can install it either through ``pip install gym``.


.. code-block:: python

    from ai_traineree.agents.dqn import DQNAgent
    from ai_traineree.env_runner import EnvRunner
    from ai_traineree.tasks import GymTask

    task = GymTask('CartPole-v1')
    agent = DQNAgent(task.state_size, task.action_size, n_steps=5)
    env_runner = EnvRunner(task, agent)

    # Learning
    scores = env_runner.run(reward_goal=100, max_episodes=300, force_new=True)

    # Check what we have learned by rendering
    env_runner.interact_episode(render=True)


Multi agent
-----------

IQL on Prison
`````````````

This example uses the Prison environment provided by the `PettingZoo <https://www.pettingzoo.ml/>`.
The *Prison* is simple environment where all agents are independent with a simple task alternatively
touch walls. To install the environment execute ``pip install pettingzoo[butterfly]``.

.. code-block:: python

    from ai_traineree.multi_agent.iql import IQLAgents
    from ai_traineree.multiagent_env_runner import MultiAgentCycleEnvRunner
    from ai_traineree.tasks import PettingZooTask
    from pettingzoo.butterfly import prison_v2 as prison

    env = prison.env(vector_observation=True)
    task = PettingZooTask(env)
    task.reset()

    config = {
        'device': 'cpu',
        'update_freq': 10,
        'batch_size': 200,
        'agent_names': env.agents,
    }
    agents = IQLAgents(task.state_size, task.action_size.n, task.num_agents, **config)

    env_runner = MultiAgentCycleEnvRunner(task, agents, max_iterations=9000, data_logger=data_logger)
    scores = env_runner.run(reward_goal=20, max_episodes=50, eps_decay=0.95, log_episode_freq=1, force_new=True)
