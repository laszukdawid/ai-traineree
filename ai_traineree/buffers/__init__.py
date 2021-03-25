import abc
import numpy as np
import torch

from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
from ai_traineree import to_list

# *Note*: Below these classes are additional imports to keep things backward compatible and easier to import.


class Experience(object):
    """
    Data type used to store experiences in experience buffers.
    """

    __must_haves = ['state', 'action', 'reward', 'next_state', 'done', 'state_idx', 'next_state_idx']
    keys = __must_haves + ['advantage', 'logprob', 'value', 'priority', 'index', 'weight']

    def __init__(self, **kwargs):

        for key in self.__must_haves:
            setattr(self, key, kwargs.pop(key, None))

        for key in self.keys:
            if key in kwargs:
                setattr(self, key, kwargs.get(key))

    def get_dict(self, serialize=False) -> Dict[str, Any]:
        if serialize:
            return {k: to_list(v) for (k, v) in self.__dict__.items() if k in self.keys}
        return {k: v for (k, v) in self.__dict__.items() if k in self.keys}


class BufferBase(abc.ABC):
    """Abstract class that defines buffer."""

    type: str
    all_data: List

    def add(self, **kwargs):
        """Add samples to the buffer."""
        raise NotImplementedError("You shouldn't see this. Look away. Or fix it.")

    def sample(self, *args, **kwargs) -> Optional[List[Experience]]:
        """Sample buffer for a set of experience."""
        raise NotImplementedError("You shouldn't see this. Look away. Or fix it.")

    def dump_buffer(self, serialize: bool=False) -> List[Dict]:
        """Return the whole buffer, e.g. for storing."""
        raise NotImplementedError("You shouldn't see this. Look away. Or fix it.")

    def load_buffer(self, buffer: List[Dict]):
        """Loads provided data into the buffer."""
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
from .nstep import NStepBuffer  # flake8: noqa
from .per import PERBuffer
from .replay import ReplayBuffer
from .rollout import RolloutBuffer
