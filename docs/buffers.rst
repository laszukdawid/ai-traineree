Buffers
=======

Basis
----------
This class is the abstraction for all buffers. In short, each buffer should support adding new samples and sampling from the buffer.
Additional classes are required for saving or resuming the whole state.
All buffers internally store data in as Experience but on sampling these are converted into torch Tensors or numpy arrays.

.. autoclass:: ai_traineree.buffers.BufferBase
    :members:

.. autoclass:: ai_traineree.Experience
    :members:

Replay Buffer
-------------

The most basic buffer. Supports uniform sampling.

.. autoclass:: ai_traineree.buffers.ReplayBuffer
    :members:
    :special-members:

Replay Experience Buffer (PER)
------------------------------

.. autoclass:: ai_traineree.buffers.PERBuffer
    :members:
    :special-members:
    
Rollout Buffer
------------------------------

.. autoclass:: ai_traineree.buffers.RolloutBuffer
    :members:
    :special-members:
    