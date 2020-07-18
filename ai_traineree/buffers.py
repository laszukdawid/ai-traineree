from ai_traineree.types import ActionType, RewardType, StateType
import numpy as np
import random
import torch

from collections import deque, namedtuple

device = DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, batch_size: int, buffer_size=10000, device=None):
        self.batch_size = batch_size
        self.buffer_size: int = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.experiance = namedtuple("exp", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.device = device if device is not None else DEVICE

    def add(self, state: StateType, action: ActionType, reward: RewardType, next_state: StateType, done: bool) -> None:
        exp = self.experiance(state, action, reward, next_state, done)
        self.memory.append(exp)

    @staticmethod
    def __convert_float(x):
        return torch.from_numpy(np.vstack(x)).float().to(DEVICE)

    @staticmethod
    def __convert_long(x):
        return torch.from_numpy(np.vstack(x)).long().to(DEVICE)

    @staticmethod
    def __convert_int(x):
        return torch.from_numpy(np.vstack(x).astype(np.uint8)).float().to(DEVICE)

    def sample(self):
        experiences = [exp for exp in random.sample(self.memory, k=self.batch_size) if exp is not None]

        states = self.__convert_float([e.state for e in experiences])
        actions = self.__convert_long([e.action for e in experiences])
        rewards = self.__convert_float([e.reward for e in experiences])
        next_states = self.__convert_float([e.next_state for e in experiences])
        dones = self.__convert_int([e.done for e in experiences])

        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        return len(self.memory)


        states = convert_float([e.state for e in experiences])
        actions = convert_long([e.action for e in experiences])
        rewards = convert_float([e.reward for e in experiences])
        next_states = convert_float([e.next_state for e in experiences])
        dones = convert_int([e.done for e in experiences])

        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        return len(self.memory)
