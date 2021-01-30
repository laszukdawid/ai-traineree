Tasks
=====

In short, a Task is a bit more than environment. Task takes an environment, e.g. CartPole,
as an input but it also handles state transformation and reward shaping.
A Task also aims to be compatible with OpenAI Gym's API. Some environments aren't compatible
and so we need to make them.

.. automodule:: ai_traineree.tasks
    :members:
