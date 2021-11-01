from collections import defaultdict, deque
from typing import Deque, Dict, Iterator, List, Optional

from ai_traineree.buffers import ReferenceBuffer
from ai_traineree.types.state import BufferState

from . import BufferBase, Experience


class RolloutBuffer(BufferBase):

    type = "Rollout"

    def __init__(self, batch_size: int, buffer_size=int(1e6), **kwargs):
        """
        A buffer that keeps and returns data in order.
        Commonly used with on-policy methods such as PPO.

        Parameters:
            batch_size (int): Maximum number of samples to return in each batch.
            buffer_size (int): Number of samples to store in the buffer.

        Keyword Arguments:
            compress_state (bool): Default False. Whether to manage memory used by states.
                Useful when states are "large" and frequently visited. Typical use case is
                dealing with images.

        """
        super().__init__()
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.data: Deque = deque()

        self._states_mng = kwargs.get("compress_state", False)
        self._states = ReferenceBuffer(buffer_size + 20)

    def __len__(self) -> int:
        return len(self.data)

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o)

    def clear(self):
        self.data.clear()

    def add(self, **kwargs):
        if self._states_mng:
            kwargs["state_idx"] = self._states.add(kwargs.pop("state"))
            if "next_state" in kwargs:
                kwargs["next_state_idx"] = self._states.add(kwargs.pop("next_state", "None"))
        self.data.append(Experience(**kwargs))

        if len(self.data) > self.buffer_size:
            drop_exp = self.data.popleft()
            if self._states_mng:
                self._states.remove(drop_exp.state_idx)
                self._states.remove(drop_exp.next_state_idx)

    def sample(self, batch_size: Optional[int] = None) -> Iterator[Dict[str, list]]:
        """
        Samples the whole buffer. Iterates all gathered data.
        Note that sampling doesn't clear the buffer.

        Returns:
            A generator that iterates over all rolled-out samples.
        """
        data = self.data.copy()
        batch_size = batch_size if batch_size is not None else self.batch_size

        while len(data):
            batch_size = min(batch_size, len(data))
            all_experiences = defaultdict(lambda: [])
            for _ in range(batch_size):
                sample = data.popleft()
                for key, value in sample.get_dict().items():
                    all_experiences[key].append(value)

            yield all_experiences

    def all_samples(self):
        all_experiences = defaultdict(lambda: [])
        for sample in self.data:
            for key, value in sample.get_dict().items():
                all_experiences[key].append(value)

        return all_experiences

    @staticmethod
    def from_state(state: BufferState):
        if state.type != RolloutBuffer.type:
            raise ValueError(f"Can only populate own type. '{RolloutBuffer.type}' != '{state.type}'")
        buffer = RolloutBuffer(batch_size=state.batch_size, buffer_size=state.buffer_size)
        if state.data:
            buffer.load_buffer(state.data)
        return buffer

    def dump_buffer(self, serialize: bool = False) -> Iterator[Dict[str, List]]:
        for data in self.data:
            yield data.get_dict(serialize=serialize)

    def load_buffer(self, buffer: List[Experience]):
        for experience in buffer:
            self.add(**experience.data)

    def seed(self, seed: int) -> None:
        pass
