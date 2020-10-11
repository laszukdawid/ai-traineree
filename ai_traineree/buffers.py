import math
import numpy as np
import random
import torch

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Sequence, Tuple
from torch import Tensor


class Experience(object):

    keys = [
        'state', 'action', 'reward', 'next_state', 'done',
        'advantage', 'logprob', 'value',
        'priority', 'index',
    ]

    def __init__(self, **kwargs):

        # for key in self.keys:
        #     if key in kwargs:
        #         self.__setattr__(key, kwargs[key])

        self.state = kwargs.get('state')
        self.action = kwargs.get('action')
        self.reward = kwargs.get('reward')
        self.next_state = kwargs.get('next_state')
        self.done = kwargs.get('done')
        self.advantage = kwargs.get('advantage')
        self.logprob = kwargs.get('logprob')
        self.value = kwargs.get('value')

        self.index = kwargs.get('index')
        self.weight = kwargs.get('weight')

    def get_dict(self) -> Dict[str, Any]:
        return dict(state=self.state, action=self.action, reward=self.reward, next_state=self.next_state, done=self.done)


class BufferBase(object):

    def add(self, **kwargs):
        raise NotImplementedError("You shouldn't see this. Look away. Or fix it.")

    def sample(self, *args, **kwargs) -> Optional[List[Experience]]:
        raise NotImplementedError("You shouldn't see this. Look away. Or fix it.")

    @staticmethod
    def convert_float(x):
        return torch.from_numpy(np.vstack(x)).float()

    @staticmethod
    def convert_long(x):
        return torch.from_numpy(np.vstack(x)).long()

    @staticmethod
    def convert_int(x):
        return torch.from_numpy(np.vstack(x).astype(np.uint8)).float()


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


class ReplayBuffer(BufferBase):

    def __init__(self, batch_size: int, buffer_size=int(1e6), device=None):
        super().__init__()
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        self.indices = range(batch_size)

        self.exp: deque = deque(maxlen=buffer_size)

        self.advantages = deque(maxlen=buffer_size)
        self.states = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.next_states = deque(maxlen=buffer_size)
        self.logprobs = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)
        self.masks = deque(maxlen=buffer_size)
        self.values = deque(maxlen=buffer_size)

    def __len__(self) -> int:
        return max(len(self.exp), len(self.states))

    def add(self, **kwargs):
        self.exp.append(Experience(**kwargs))

    def add_sars(self, *, state=None, action=None, reward=None, next_state=None, done=None) -> None:
        """Adds (State, Actiom, Reward, State) to the buffer. Expects these arguments to be named properties."""
        self.exp.append(Experience(state=state, action=action, reward=reward, next_state=next_state, done=done))

    def sample(self) -> Dict[str, List]:
        all_experiences = defaultdict(lambda: [])
        sampled_exp = random.sample(self.exp, self.batch_size)
        for exp in sampled_exp:
            for (key, val) in exp.__dict__.items():
                all_experiences[key].append(val)
        return all_experiences

    def sample_sars(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for exp in random.sample(self.exp, self.batch_size):
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(exp.next_state)
            dones.append(exp.done)

        states = self.convert_float(states).to(self.device)
        actions = self.convert_float(actions).to(self.device)
        rewards = self.convert_float(rewards).to(self.device)
        next_states = self.convert_float(next_states).to(self.device)
        dones = self.convert_int(dones).to(self.device)

        return (states, actions, rewards, next_states, dones)


class PERBuffer(BufferBase):
    """Prioritized Experience Replay

    Based on "Prioritized Experience Replay" (2016) T. Shaul, J. Quan, I. Antonoglou, D. Silver.
    https://arxiv.org/pdf/1511.05952.pdf
    """

    def __init__(self, batch_size, buffer_size: int=int(1e6), alpha=0.05, device=None):
        super(PERBuffer, self).__init__()
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        self.tree = SumTree(buffer_size)
        self.alpha: float = alpha
        self.__default_weights = np.ones(self.batch_size)/self.buffer_size

        self.tiny_offset: float = 0.05

    def __len__(self) -> int:
        return len(self.tree)

    def add(self, *, priority: float=0, **kwargs):
        priority += self.tiny_offset
        self.tree.insert(kwargs, pow(priority, self.alpha))

    def add_sars(self, **kwargs):
        self.add(**kwargs)

    def sample_list(self, beta: float=1, **kwargs) -> Optional[List[Experience]]:
        """The method return samples randomly without duplicates"""
        if len(self.tree) < self.batch_size:
            return None

        samples = []
        experiences = []
        indices = []
        weights = self.__default_weights.copy()
        priorities = []
        for k in range(self.batch_size):
            r = random.random()
            data, priority, index = self.tree.find(r)
            priorities.append(priority)
            weights[k] = pow(weights[k]/priority, beta) if priority > 1e-16 else 0
            indices.append(index)
            samples.append(data)
            self.priority_update([index], [0])  # To avoid duplicating

        self.priority_update(indices, priorities)  # Revert priorities
        weights = weights / max(weights)
        for k in range(self.batch_size):
            experience = Experience(**samples[k], index=indices[k], weight=weights[k])
            experiences.append(experience)

        return experiences

    def sample(self, beta: float=0.5) -> Optional[Dict[str, List]]:
        all_experiences = defaultdict(lambda: [])
        sampled_exp = self.sample_list(beta=beta)
        if sampled_exp is None:
            return None

        for exp in sampled_exp:
            for (key, val) in exp.__dict__.items():
                all_experiences[key].append(val)
        return all_experiences

    def sample_sars(self) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        raw_samples = self.sample()
        if raw_samples is None:
            return None

        experiences = raw_samples
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for exp in experiences:
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(exp.next_state)
            dones.append(exp.done)

        states = self.convert_float(states)
        actions = self.convert_float(actions)
        rewards = self.convert_float(rewards)
        next_states = self.convert_float(next_states)
        dones = self.convert_int(dones)

        return states, actions, rewards, next_states, dones

    def priority_update(self, indices: Sequence[int], priorities: Sequence[float]) -> None:
        """Updates prioprities for elements on provided indices."""
        for i, p in zip(indices, priorities):
            self.tree.weight_update(i, math.pow(p, self.alpha))

    def reset_alpha(self, alpha: float):
        """Resets the alpha wegith (p^alpha)"""
        tree_len = len(self.tree)
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [pow(self.tree[i], -old_alpha) for i in range(tree_len)]
        self.priority_update(range(tree_len), priorities)


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
        self.data = [None for i in range(self.leafs_num)]
        self.size = 0
        self.cursor = 0

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index) -> float:
        if isinstance(index, slice):
            return self.tree[self.leaf_offset:][index]
        return self.tree[self.leaf_offset + index]

    def insert(self, data, weight):
        index = self.cursor
        self.cursor = (self.cursor+1) % self.leafs_num
        self.size = min(self.size+1, self.leafs_num)

        self.data[index] = data
        self.weight_update(index, weight)

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
            return self.data[index - self.leaf_offset], self.tree[index], index - self.leaf_offset

        left_idx = 2*index + 1
        left_weight = self.tree[left_idx]

        if weight <= left_weight:
            return self._find(weight, left_idx)
        else:
            return self._find(weight - left_weight, 2*(index+1))

    def __str__(self):
        s = ""
        for k in range(self.tree_height):
            s += " ".join([str(v) for v in self.tree[2**k-1:2**(k+1)-1]])
            s += "\n"
        return s
