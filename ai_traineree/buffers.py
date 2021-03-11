import abc
import math
import numpy as np
import random
import torch

from ai_traineree import to_list
from collections import defaultdict, deque
from copy import copy
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union


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


class NStepBuffer(BufferBase):
    def __init__(self, n_steps: int, gamma: float):
        super().__init__()
        self.gamma = gamma
        self.n_steps = n_steps
        self.n_gammas = [gamma**i for i in range(1, n_steps+1)]

        self.buffer = []
        self.reward_buffer = []
        self.done_buffer = []

    def __len__(self):
        return len(self.buffer)

    @property
    def available(self):
        return len(self.buffer) >= self.n_steps

    def add(self, **kwargs):
        self.buffer.append(Experience(**kwargs))

    def get(self) -> Experience:
        current_exp = self.buffer.pop(0)

        for (idx, exp) in enumerate(self.buffer):
            if exp.done[0]:
                break
            current_exp.reward[0] += self.n_gammas[idx]*exp.reward[0]
        return current_exp


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


class ReplayBuffer(BufferBase):

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

    def dump_buffer(self, serialize: bool=False) -> Iterator[Dict[str, List]]:
        for exp in self.exp:
            yield exp.get_dict(serialize=serialize)

    def load_buffer(self, buffer: List[Dict[str, List]]):
        for experience in buffer:
            self.add(**experience)


class RolloutBuffer(BufferBase):

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
        self.exp = deque()

        self._states_mng = kwargs.get('compress_state', False)
        self._states = ReferenceBuffer(buffer_size + 20)

    def __len__(self) -> int:
        return len(self.exp)

    def clear(self):
        self.exp.clear()

    def add(self, **kwargs):
        if self._states_mng:
            kwargs['state_idx'] = self._states.add(kwargs.pop("state"))
            if "next_state" in kwargs:
                kwargs['next_state_idx'] = self._states.add(kwargs.pop("next_state", "None"))
        self.exp.append(Experience(**kwargs))

        if len(self.exp) > self.buffer_size:
            drop_exp = self.exp.popleft()
            if self._states_mng:
                self._states.remove(drop_exp.state_idx)
                self._states.remove(drop_exp.next_state_idx)

    def sample(self, batch_size: Optional[int]=None) -> Iterator[Dict[str, list]]:
        """
        Samples the whole buffer. Iterates all gathered data.
        Note that sampling doesn't clear the buffer.

        Returns:
            A generator that iterates over all rolled-out samples.
        """
        exp = self.exp.copy()
        batch_size = batch_size if batch_size is not None else self.batch_size

        while len(exp):
            batch_size = min(batch_size, len(exp))
            all_experiences = defaultdict(lambda: [])
            for _ in range(batch_size):
                sample = exp.popleft()
                for key, value in sample.get_dict().items():
                    all_experiences[key].append(value)

            yield all_experiences

    def all_samples(self):
        all_experiences = defaultdict(lambda: [])
        for sample in self.exp:
            for key, value in sample.get_dict().items():
                all_experiences[key].append(value)

        return all_experiences

    def dump_buffer(self, serialize: bool=False) -> Iterator[Dict[str, List]]:
        for exp in self.exp:
            yield exp.get_dict(serialize=serialize)

    def load_buffer(self, buffer: List[Dict[str, List]]):
        for experience in buffer:
            self.add(**experience)

    def seed(self, seed: int) -> None:
        pass


class PERBuffer(BufferBase):
    """Prioritized Experience Replay

    A buffer that holds previously seen sets of transitions, or memories.
    Prioritization in the name means that each transition has some value (priority)
    which refers to the probability of sampling that transition.
    In short, the larger priority value the higher chances of sampling associated samples.
    Often these priority values are related to the error calculated when learning from
    that associated sample. In such cases, sampling from the buffer will more often provide
    values that are troublesome.

    Based on "Prioritized Experience Replay" (2016) T. Shaul, J. Quan, I. Antonoglou, D. Silver.
    https://arxiv.org/pdf/1511.05952.pdf

    """

    def __init__(self, batch_size: int, buffer_size: int=int(1e6), alpha=0.5, device=None, **kwargs):
        """
        Parameters:
            batch_size (int): Number of samples to return on sampling.
            buffer_size (int): Maximum number of samples to store. Default: 10^6.
            alpha (float): Optional (default: 0.5).
                Power factor for priorities making the sampling prob ~priority^alpha.
            compress_state (bool): Optional (default: False).
                Whether manage memory used by states. Useful when states are "large".
                Improves memory usage but has a significant performance penalty.
            seed (int): Optional (default None). Set seed for the random number generator.

        """
        super(PERBuffer, self).__init__()
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        self.tree = SumTree(buffer_size)
        self.alpha: float = alpha
        self.__default_weights = np.ones(self.batch_size)/self.buffer_size
        self._rng = random.Random(kwargs.get('seed'))

        self.tiny_offset: float = 0.05

        self._states_mng = kwargs.get('compress_state', False)
        self._states = ReferenceBuffer(buffer_size + 20)

    def __len__(self) -> int:
        return len(self.tree)

    def seed(self, seed):
        return seed

    def add(self, *, priority: float=0, **kwargs):
        priority += self.tiny_offset
        if self._states_mng:
            kwargs['state_idx'] = self._states.add(kwargs.pop("state"))
            if "next_state" in kwargs:
                kwargs['next_state_idx'] = self._states.add(kwargs.pop("next_state"))
        old_data = self.tree.insert(kwargs, pow(priority, self.alpha))

        if len(self.tree) >= self.buffer_size and self._states_mng and old_data is not None:
            self._states.remove(old_data['state_idx'])
            self._states.remove(old_data['next_state_idx'])

    def _sample_list(self, beta: float=1, **kwargs) -> List[Experience]:
        """The method return samples randomly without duplicates"""
        if len(self.tree) < self.batch_size:
            return []

        samples = []
        experiences = []
        indices = []
        weights = self.__default_weights.copy()
        priorities = []
        for k in range(self.batch_size):
            r = self._rng.random()
            data, priority, index = self.tree.find(r)
            priorities.append(priority)
            weights[k] = pow(weights[k]/priority, beta) if priority > 1e-16 else 0
            indices.append(index)
            self.priority_update([index], [0])  # To avoid duplicating
            samples.append(data)

        self.priority_update(indices, priorities)  # Revert priorities
        weights = weights / max(weights)
        for k in range(self.batch_size):
            experience = Experience(**samples[k], index=indices[k], weight=weights[k])
            if self._states_mng:
                experience.state = self._states.get(experience.state_idx)
                experience.next_state = self._states.get(experience.next_state_idx)
            experiences.append(experience)

        return experiences

    def sample(self, beta: float=0.5) -> Optional[Dict[str, List]]:
        all_experiences = defaultdict(lambda: [])
        sampled_exp = self._sample_list(beta=beta)
        if len(sampled_exp) == 0:
            return None

        for exp in sampled_exp:
            for key in exp.__dict__.keys():
                if self._states_mng and (key == 'state' or key == 'next_state'):
                    value = self._states.get(getattr(exp, key + '_idx'))
                else:
                    value = getattr(exp, key)
                all_experiences[key].append(value)
        return all_experiences

    def priority_update(self, indices: Sequence[int], priorities: torch.Tensor) -> None:
        """Updates prioprities for elements on provided indices."""
        for i, p in zip(indices, priorities):
            self.tree.weight_update(i, math.pow(p, self.alpha))

    def reset_alpha(self, alpha: float):
        """Resets the alpha wegith (p^alpha)"""
        tree_len = len(self.tree)
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [pow(self.tree[i], -old_alpha) for i in range(tree_len)]
        self.priority_update(range(tree_len), priorities)

    def dump_buffer(self, serialize: bool=False) -> Iterator[Dict[str, List]]:
        for exp in self.tree.data[:len(self.tree)]:
            yield Experience(**exp).get_dict(serialize=serialize)

    def load_buffer(self, buffer: List[Dict[str, List]]):
        for experience in buffer:
            self.add(**experience)


class SumTree(object):
    """Binary tree that is a SumTree.
    Each level contains a sum of all its nodes.
    """
    def __init__(self, leafs_num):
        """Expects `max_size` which is the number of leaf nodes."""
        self.leafs_num = leafs_num
        self.tree_height = math.ceil(math.log(leafs_num, 2)) + 1
        self.leaf_offset = 2**(self.tree_height-1) - 1
        self.tree_size = 2**self.tree_height - 1
        self.tree = np.zeros(self.tree_size)
        self.data: List[Optional[Dict]] = [None for i in range(self.leafs_num)]
        self.size = 0
        self.cursor = 0

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index) -> float:
        if isinstance(index, slice):
            return self.tree[self.leaf_offset:][index]
        return self.tree[self.leaf_offset + index]

    def insert(self, data, weight) -> Any:
        """Returns `data` of element that was removed"""
        index = self.cursor
        self.cursor = (self.cursor+1) % self.leafs_num
        self.size = min(self.size+1, self.leafs_num)

        old_data = copy(self.data[index])
        self.data[index] = data
        self.weight_update(index, weight)
        return old_data

    def weight_update(self, index, weight):
        tree_index = self.leaf_offset + index
        diff = weight - self.tree[tree_index]
        self._tree_update(tree_index, diff)

    def _tree_update(self, tindex, diff):
        self.tree[tindex] += diff
        if tindex != 0:
            tindex = (tindex-1) // 2
            self._tree_update(tindex, diff)

    def find(self, weight) -> Tuple[Any, float, int]:
        """Returns (data, weight, index)"""
        assert 0 <= weight <= 1, "Expecting weight to be sampling weight [0, 1]"
        return self._find(weight*self.tree[0], 0)

    def _find(self, weight, index) -> Tuple[Any, float, int]:
        if self.leaf_offset <= index:  # Moved to the leaf layer
            return self.data[min(index - self.leaf_offset, self.leafs_num-1)], self.tree[index], index - self.leaf_offset

        left_idx = 2*index + 1
        left_weight = self.tree[left_idx]

        if weight <= left_weight:
            return self._find(weight, left_idx)
        else:
            return self._find(weight - left_weight, 2*(index+1))

    def get_n_first_nodes(self, n):
        return self.data[:n]
