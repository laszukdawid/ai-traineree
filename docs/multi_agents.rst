Multi agents
============

Usage of "agents" in this case could be a bit misleading. Here are entitites or algorithms
that understand how to organize internal agents to get better in interacting with the environment.

The distinction between these and many individual is that some interaction between agents is assumed.
It isn't a single agent that tries to do something in the environment and could consider other agents
as part of the environment. Typical cases for multi agents is when they need to achieve a common goal.
Consider cooperative games like not letting a ball fall on the ground, or team sports where one team
tries to capture a flag and the other tries to stop them.

MADDPG
------

.. autoclass:: ai_traineree.multi_agents.maddpg.MADDPGAgent
    :members:
    :undoc-members:

IQL
---

.. autoclass:: ai_traineree.multi_agents.iql.IQLAgents
    :members:
    :undoc-members:

