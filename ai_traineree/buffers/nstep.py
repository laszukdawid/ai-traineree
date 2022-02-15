from collections import deque
from typing import Deque, List

from ai_traineree.types import BufferState

from . import BufferBase, Experience


class NStepBuffer(BufferBase):

    type = "NStep"
    gamma: float

    def __init__(self, n_steps: int, gamma: float):
        super().__init__()
        self.gamma = gamma
        self.n_steps = n_steps
        self.n_gammas = [gamma**i for i in range(1, n_steps + 1)]

        self.data: Deque = deque(maxlen=n_steps)

        # For consistency with other buffers
        self.buffer_size = n_steps
        self.batch_size = 1

    def __len__(self):
        return len(self.data)

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and isinstance(o, type(self)) and self.gamma == o.gamma

    @property
    def available(self):
        return len(self.data) >= self.n_steps

    def clear(self):
        self.data = deque(maxlen=self.n_steps)

    def add(self, **kwargs):
        self.data.append(Experience(**kwargs))

    def get(self) -> Experience:
        # current_exp = self.data.pop(0)
        current_exp = self.data.popleft()

        for (idx, exp) in enumerate(self.data):
            if exp.done[0]:
                break
            current_exp.reward[0] += self.n_gammas[idx] * exp.reward[0]
        return current_exp

    def get_state(self, include_data: bool = True) -> BufferState:
        state = super().get_state(include_data=include_data)
        state.extra = dict(gamma=self.gamma)
        return state

    @staticmethod
    def from_state(state: BufferState):
        if state.type != NStepBuffer.type:
            raise ValueError(f"Can only populate own type. '{NStepBuffer.type}' != '{state.type}'")
        if state.batch_size != 1:
            raise ValueError(
                f"Provided batch_size={state.batch_size} for {state.type}. Only batch_size=1 is currently supported."
            )
        buffer = NStepBuffer(n_steps=state.buffer_size, gamma=state.extra.get("gamma", 1.0))
        if state.data:
            buffer.load_buffer(state.data)
        return buffer

    def load_buffer(self, buffer: List[Experience]):
        for experience in buffer:
            self.add(**experience.data)
