import random

from typing import Dict, Iterator, List, Optional, Sequence

from . import BufferBase, Experience
from ai_traineree.buffers import ReferenceBuffer


class ReplayBuffer(BufferBase):

    type = "Replay"
    keys = ["states", "actions", "rewards", "next_states", "dones"]

    def __init__(self, batch_size: int, buffer_size=int(1e6), device=None, **kwargs):
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
        self.device = device
        self.indices = range(batch_size)
        self.exp: List = []

        self._states_mng = kwargs.get('compress_state', False)
        self._states = ReferenceBuffer(buffer_size + 20)
        self._rng = random.Random(kwargs.get('seed'))

    def __len__(self) -> int:
        return len(self.exp)

    def seed(self, seed: int):
        self._rng = random.Random(seed)

    def clear(self):
        self.exp = []

    def add(self, **kwargs):
        if self._states_mng:
            kwargs['state_idx'] = self._states.add(kwargs.pop("state"))
            if "next_state" in kwargs:
                kwargs['next_state_idx'] = self._states.add(kwargs.pop("next_state", "None"))
        self.exp.append(Experience(**kwargs))

        if len(self.exp) > self.buffer_size:
            drop_exp = self.exp.pop(0)
            if self._states_mng:
                self._states.remove(drop_exp.state_idx)
                self._states.remove(drop_exp.next_state_idx)

    def sample(self, keys: Optional[Sequence[str]]=None) -> Dict[str, List]:
        """
        Parameters:
            keys: A list of keys which limit the return.
                If nothing is provided, all keys are returned.

        Returns:
            Returns all values for asked keys.
        """
        sampled_exp: List[Experience] = self._rng.sample(self.exp, self.batch_size)
        keys = keys if keys is not None else list(self.exp[0].__dict__.keys())
        all_experiences = {k: [] for k in keys}
        for exp in sampled_exp:
            for key in keys:
                if self._states_mng and (key == 'state' or key == 'next_state'):
                    value = self._states.get(getattr(exp, key + '_idx'))
                else:
                    value = getattr(exp, key)

                all_experiences[key].append(value)
        return all_experiences

    @property
    def all_data(self) -> List:
        return self.exp

    def dump_buffer(self, serialize: bool=False) -> Iterator[Dict[str, List]]:
        for exp in self.exp:
            yield exp.get_dict(serialize=serialize)

    def load_buffer(self, buffer: List[Dict[str, List]]):
        for experience in buffer:
            self.add(**experience)
