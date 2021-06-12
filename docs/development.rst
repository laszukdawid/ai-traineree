Development
===========

Philosophy
----------

* Agents are independent from environment. No interaction is forced.
* All agents should have the same concise APIs.
* Modular components but simplicity over forced modularity.

Concepts
--------

State vs Observation vs Features
````````````````````````````````
State is an objective information about the enviroment. It is from external entity's point of view.
Access to states isn't guaranteed even if one has full control over the environment.

Observation is from agent's perspective. Its domain is defined by agent's senses and values depend on agent's state, e.g. position.

Features are context dependent but generally relate to some output of a transformation.
We can transform observation to a different space, e.g. projecting camera RGB image into an embedding vector, or modify values, e.g. normalize tensor.

Example:
Considering basketball game as an environment.
A spectator is the one who might have access to the state information.
In this case, a state would consist of all players' positions, ball possession, time and score.
However, decide being able to see everything they wouldn't know whether any player is feeling bad or some places on the field have draft.
An agent, in this situation, is a single player.
Everything that they see and know is their observation.
Although they might be able to deduce position of all players, it will often happen that some players will be behind others
or they will be looking in a different direction.
Similar to spectator they don't know about other players stamina levels but they know theirs which also has an impact on the play.
Their physical state and internal thoughts are features.


Code-wise, state is the output of enviroment. Observation is what an agent can get and deals with on input.
Feature is anything that goes through any transfomrations, e.g. bodies and heads.
A specific case of a feature is an action.
