from aitraineree.buffers import BufferBase
from aitraineree.buffers.nstep import NStepBuffer
from aitraineree.buffers.per import PERBuffer
from aitraineree.buffers.replay import ReplayBuffer
from aitraineree.buffers.rollout import RolloutBuffer
from aitraineree.types import BufferState


class BufferFactory:
    @staticmethod
    def from_state(state: BufferState) -> BufferBase:
        if state.type == ReplayBuffer.type:
            return ReplayBuffer.from_state(state)
        elif state.type == PERBuffer.type:
            return PERBuffer.from_state(state)
        elif state.type == NStepBuffer.type:
            return NStepBuffer.from_state(state)
        elif state.type == RolloutBuffer.type:
            return RolloutBuffer.from_state(state)
        else:
            raise ValueError(f"Buffer state contains unsupported buffer type: '{state.type}'")
