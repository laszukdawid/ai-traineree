from ai_traineree.types import BufferState
from ai_traineree.buffers import BufferBase, ReplayBuffer, PERBuffer, NStepBuffer, RolloutBuffer


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
