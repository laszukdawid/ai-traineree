import random
from typing import Iterator, Sequence

from ai_traineree.buffers import ReferenceBuffer
from ai_traineree.types.state import BufferState

from . import BufferBase, Experience


class ReplayBuffer(BufferBase):
    type = "Replay"
    keys = ["states", "actions", "rewards", "next_states", "dones"]

    def __init__(self, batch_size: int, buffer_size=int(1e6), **kwargs):
        """
        Parameters:
            compress_state: bool (default: False)
                Whether manage memory used by states. Useful when states are "large".
                Improves memory usage but has a significant performance penalty.
            seed: int (default: None)
                Set seed for the random number generator.
        """
        super().__init__()
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.indices = range(batch_size)
        self.data: list[Experience] = []

        self._states_mng = kwargs.get("compress_state", False)
        self._states = ReferenceBuffer(buffer_size + 20)
        self._rng = random.Random(kwargs.get("seed"))

    def __len__(self) -> int:
        return len(self.data)

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and isinstance(o, type(self))

    def seed(self, seed: int):
        self._rng = random.Random(seed)

    def clear(self):
        """Removes all data from the buffer"""
        self.data = []

    def add(self, **kwargs):
        if self._states_mng:
            kwargs["state_idx"] = self._states.add(kwargs.pop("state"))
            if "next_state" in kwargs:
                kwargs["next_state_idx"] = self._states.add(kwargs.pop("next_state", "None"))
        self.data.append(Experience(**kwargs))

        if len(self.data) > self.buffer_size:
            drop_exp = self.data.pop(0)
            if self._states_mng:
                self._states.remove(drop_exp.state_idx)
                self._states.remove(drop_exp.next_state_idx)

    def sample(self, keys: Sequence[str] | None = None) -> dict[str, list]:
        """
        Parameters:
            keys: A list of keys which limit the return.
                If nothing is provided, all keys are returned.

        Returns:
            Returns all values for asked keys.
        """
        sampled_exp: list[Experience] = self._rng.sample(self.data, self.batch_size)
        keys = keys if keys is not None else list(self.data[0].__dict__.keys())
        all_experiences = {k: [] for k in keys}
        for data in sampled_exp:
            for key in keys:
                if self._states_mng and (key == "state" or key == "next_state"):
                    value = self._states.get(getattr(data, key + "_idx"))
                else:
                    value = getattr(data, key)

                all_experiences[key].append(value)
        return all_experiences

    @staticmethod
    def from_state(state: BufferState):
        if state.type != ReplayBuffer.type:
            raise ValueError(f"Can only populate own type. '{ReplayBuffer.type}' != '{state.type}'")
        buffer = ReplayBuffer(batch_size=int(state.batch_size), buffer_size=int(state.buffer_size))
        if state.data:
            buffer.load_buffer(state.data)
        return buffer

    def dump_buffer(self, serialize: bool = False) -> Iterator[dict[str, list]]:
        for data in self.data:
            yield data.get_dict(serialize=serialize)

    def load_buffer(self, buffer: list[Experience]):
        for experience in buffer:
            # self.add(**experience)
            self.add(**experience.data)
