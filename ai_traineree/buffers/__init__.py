import abc
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from ai_traineree.types import BufferState
from ai_traineree.types.experience import Experience

# *Note*: Below classes are additional imports to keep things backward compatible and easier to import.


class BufferBase(abc.ABC):
    """Abstract class that defines buffer."""

    type: str
    data: Sequence  # Experience
    batch_size: int
    buffer_size: int

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, type(self))
            and self.batch_size == o.batch_size
            and self.buffer_size == o.buffer_size
            and self.data == o.data
        )

    def __len__(self):
        return len(self.data)

    def add(self, **kwargs):
        """Add samples to the buffer."""
        raise NotImplementedError("You shouldn't see this. Look away. Or fix it.")

    def sample(self, *args, **kwargs) -> Optional[List[Experience]]:
        """Sample buffer for a set of experience."""
        raise NotImplementedError("You shouldn't see this. Look away. Or fix it.")

    def dump_buffer(self, serialize: bool = False) -> List[Dict]:
        """Return the whole buffer, e.g. for storing."""
        raise NotImplementedError("You shouldn't see this. Look away. Or fix it.")

    def load_buffer(self, buffer: List[Experience]) -> None:
        """Loads provided data into the buffer."""
        raise NotImplementedError("You shouldn't see this. Look away. Or fix it.")

    def get_state(self, include_data: bool = True) -> BufferState:
        state = BufferState(type=self.type, buffer_size=self.buffer_size, batch_size=self.batch_size)
        if len(self.data) and include_data:
            # state.data = [d.data for d in self.data]  # In case we want to serialize
            state.data = self.data
        return state

    def seed(self, seed: int) -> None:
        raise NotImplementedError("You shouldn't see this. Look away. Or fix it.")


class ReferenceBuffer(object):
    def __init__(self, buffer_size: int):
        self.buffer = dict()
        self.counter = defaultdict(int)
        self.buffer_size = buffer_size

    def __len__(self) -> int:
        return len(self.buffer)

    @staticmethod
    def _hash_element(el) -> Union[int, str]:
        if isinstance(el, np.ndarray):
            return hash(el.data.tobytes())
        elif isinstance(el, torch.Tensor):
            return hash(str(el))
        else:
            return str(el)

    def add(self, el) -> Union[int, str]:
        idx = self._hash_element(el)
        self.counter[idx] += 1
        if self.counter[idx] < 2:
            self.buffer[idx] = el
        return idx

    def get(self, idx: Union[int, str]):
        return self.buffer[idx]

    def remove(self, idx: str):
        self.counter[idx] -= 1
        if self.counter[idx] < 1:
            self.buffer.pop(idx, None)
            del self.counter[idx]


# Imports to keep things easier accessible and in tact with previous version
from .nstep import NStepBuffer  # noqa
from .per import PERBuffer  # noqa
from .replay import ReplayBuffer  # noqa
from .rollout import RolloutBuffer  # noqa
