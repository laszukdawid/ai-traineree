import copy
import math
import random
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy

from ai_traineree.types.state import BufferState

from . import BufferBase, Experience, ReferenceBuffer


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

    type = "PER"

    def __init__(self, batch_size: int, buffer_size: int = int(1e6), alpha=0.5, device=None, **kwargs):
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
        self.__default_weights = numpy.ones(self.batch_size) / self.buffer_size
        self._rng = random.Random(kwargs.get("seed"))

        self.tiny_offset: float = 0.05

        self._states_mng = kwargs.get("compress_state", False)
        self._states = ReferenceBuffer(buffer_size + 20)

    def __len__(self) -> int:
        return len(self.tree)

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and isinstance(o, type(self))

    def seed(self, seed: int):
        self._rng = random.Random(seed)
        return seed

    @property
    def data(self):
        # TODO @dawid: update to SumTree so that it return proper data
        return list(filter(lambda x: x is not None, self.tree.data))

    def add(self, *, priority: float = 0, **kwargs):
        priority += self.tiny_offset
        if self._states_mng:
            kwargs["state_idx"] = self._states.add(kwargs.pop("state"))
            if "next_state" in kwargs:
                kwargs["next_state_idx"] = self._states.add(kwargs.pop("next_state"))
        # old_data = self.tree.insert(kwargs, pow(priority, self.alpha))
        old_data = self.tree.insert(Experience(**kwargs), pow(priority, self.alpha))

        if len(self.tree) >= self.buffer_size and self._states_mng and old_data is not None:
            self._states.remove(old_data["state_idx"])
            self._states.remove(old_data["next_state_idx"])

    def _sample_list(self, beta: float = 1, **kwargs) -> List[Experience]:
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
            weights[k] = pow(weights[k] / priority, beta) if priority > 1e-16 else 0
            indices.append(index)
            self.priority_update([index], [0])  # To avoid duplicating
            samples.append(data)

        self.priority_update(indices, priorities)  # Revert priorities
        weights = weights / max(weights)

        for (experience, weight, index) in zip(samples, weights, indices):
            experience.weight = weight
            experience.index = index
            if self._states_mng:
                experience.state = self._states.get(experience.state_idx)
                experience.next_state = self._states.get(experience.next_state_idx)
            experiences.append(experience)

        return experiences

    def sample(self, beta: float = 0.5) -> Optional[Dict[str, List]]:
        all_experiences = defaultdict(lambda: [])
        sampled_exp = self._sample_list(beta=beta)
        if len(sampled_exp) == 0:
            return None

        for exp in sampled_exp:
            for key in exp.__dict__.keys():
                if self._states_mng and (key == "state" or key == "next_state"):
                    value = self._states.get(getattr(exp, key + "_idx"))
                else:
                    value = getattr(exp, key)
                all_experiences[key].append(value)
        return all_experiences

    def priority_update(self, indices: Sequence[int], priorities: List) -> None:
        """Updates prioprities for elements on provided indices."""
        for i, p in zip(indices, priorities):
            self.tree.weight_update(i, math.pow(p, self.alpha))

    def reset_alpha(self, alpha: float):
        """Resets the alpha wegith (p^alpha)"""
        tree_len = len(self.tree)
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [pow(self.tree[i], -old_alpha) for i in range(tree_len)]
        self.priority_update(range(tree_len), priorities)

    @staticmethod
    def from_state(state: BufferState):
        if state.type != PERBuffer.type:
            raise ValueError(f"Can only populate own type. '{PERBuffer.type}' != '{state.type}'")
        buffer = PERBuffer(batch_size=int(state.batch_size), buffer_size=int(state.buffer_size))

        # TODO: Populate whole tree
        if state.data:
            buffer.load_buffer(state.data)
        return buffer

    def dump_buffer(self, serialize: bool = False) -> Iterator[Dict[str, List]]:
        for exp in self.tree.data[: len(self.tree)]:
            # yield Experience(**exp).get_dict(serialize=serialize)
            yield exp.get_dict(serialize=serialize)

    def load_buffer(self, buffer: List[Experience]):
        for experience in buffer:
            self.add(**experience.data)


class SumTree(object):
    """SumTree

    A binary tree where each level contains sum of its children nodes.
    """

    def __init__(self, leafs_num: int):
        """
        Parameters:
            leafs_num (int): Number of leaf nodes.

        """
        self.leafs_num = leafs_num
        self.tree_height = math.ceil(math.log(leafs_num, 2)) + 1
        self.leaf_offset = 2 ** (self.tree_height - 1) - 1
        self.tree_size = 2 ** self.tree_height - 1
        self.tree = numpy.zeros(self.tree_size)
        self.data: List[Optional[Dict]] = [None for i in range(self.leafs_num)]
        self.size = 0
        self.cursor = 0

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index) -> float:
        if isinstance(index, slice):
            return self.tree[self.leaf_offset :][index]
        return self.tree[self.leaf_offset + index]

    def insert(self, data, weight) -> Any:
        """Returns `data` of element that was removed"""
        index = self.cursor
        self.cursor = (self.cursor + 1) % self.leafs_num
        self.size = min(self.size + 1, self.leafs_num)

        old_data = copy.copy(self.data[index])
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
            tindex = (tindex - 1) // 2
            self._tree_update(tindex, diff)

    def find(self, weight) -> Tuple[Any, float, int]:
        """Returns (data, weight, index)"""
        assert 0 <= weight <= 1, "Expecting weight to be sampling weight [0, 1]"
        return self._find(weight * self.tree[0], 0)

    def _find(self, weight, index) -> Tuple[Any, float, int]:
        if self.leaf_offset <= index:  # Moved to the leaf layer
            return (
                self.data[min(index - self.leaf_offset, self.leafs_num - 1)],
                self.tree[index],
                index - self.leaf_offset,
            )

        left_idx = 2 * index + 1
        left_weight = self.tree[left_idx]

        if weight <= left_weight:
            return self._find(weight, left_idx)
        else:
            return self._find(weight - left_weight, 2 * (index + 1))

    def get_n_first_nodes(self, n):
        return self.data[:n]
