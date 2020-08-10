from ai_traineree.types import ActionType, RewardType, StateType
import numpy as np
import random
import torch

from collections import deque

device = DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:

    def __init__(self, batch_size: int, buffer_size=10000, device=None):
        self.batch_size = batch_size
        self.buffer_size: int = buffer_size
        self.indices = range(batch_size)

        self.advantages = deque(maxlen=buffer_size)
        self.states = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.next_states = deque(maxlen=buffer_size)
        self.logprobs = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)
        self.masks = deque(maxlen=buffer_size)
        self.values = deque(maxlen=buffer_size)

        self.device = device if device is not None else DEVICE

    def __len__(self) -> int:
        return len(self.states)

    def prepare_batch(self) -> None:
        self.indices = random.sample(range(len(self.states)), self.batch_size)

    def __sample(self, prop):
        return [prop[idx] for idx in self.indices]

    def add_sars(self, state: StateType, action: ActionType, reward: RewardType, next_state: StateType, done: bool) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def add_advantage(self, advantage):
        self.advantages.append(advantage)

    def add_value(self, value):
        self.values.append(value)

    def add_logprob(self, logprob):
        self.logprobs.append(logprob)

    def add_state(self, state):
        self.states.append(state)

    def add_action(self, action):
        self.actions.append(action)

    def add_next_state(self, next_state):
        self.next_states.append(next_state)

    def add_reward(self, reward):
        self.rewards.append(reward)

    def add_done(self, done):
        self.dones.append(done)
        self.masks.append(1-done)

    def sample_advantages(self):
        return self.__sample(self.advantages)

    def sample_values(self):
        return self.__sample(self.values)

    def sample_logprobs(self):
        return self.__sample(self.logprobs)

    def sample_states(self):
        return self.__sample(self.states)

    def sample_actions(self):
        return self.__sample(self.actions)

    def sample_rewards(self):
        return self.__sample(self.rewards)

    def sample_next_states(self):
        return self.__sample(self.next_states)

    def sample_dones(self):
        return self.__sample(self.dones)

    def sample_masks(self):
        return self.__sample(self.masks)

    @staticmethod
    def __convert_float(x):
        return torch.from_numpy(np.vstack(x)).float().to(DEVICE)

    @staticmethod
    def __convert_long(x):
        return torch.from_numpy(np.vstack(x)).long().to(DEVICE)

    @staticmethod
    def __convert_int(x):
        return torch.from_numpy(np.vstack(x).astype(np.uint8)).float().to(DEVICE)

    def sample_sars(self):
        self.prepare_batch()
        states = self.__convert_float(self.sample_states())
        actions = self.__convert_long(self.sample_actions())
        rewards = self.__convert_float(self.sample_rewards())
        next_states = self.__convert_float(self.sample_next_states())
        dones = self.__convert_int(self.sample_dones())

        return (states, actions, rewards, next_states, dones)
