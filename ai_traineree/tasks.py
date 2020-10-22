import gym
from ai_traineree.types import ActionType, StateType, TaskType

from typing import Callable, Optional, Tuple


class GymTask(TaskType):
    def __init__(self, env_name: str, state_transform: Optional[Callable]=None, reward_transform: Optional[Callable]=None, can_render=True):

        self.name = env_name
        self.env = gym.make(env_name)
        self.can_render = can_render
        self.is_discrete = "Discrete" in str(type(self.env.action_space))

        state_shape = self.env.observation_space.shape
        self.state_size = state_shape[0] if len(state_shape) == 1 else state_shape
        self.action_size = self.__determine_action_size(self.env.action_space)
        self.state_transform = state_transform
        self.reward_transform = reward_transform

    @staticmethod
    def __determine_action_size(action_space):
        if "Discrete" in str(type(action_space)):
            return action_space.n
        else:
            return sum(action_space.shape)

    @property
    def actual_state_size(self):
        state = self.reset()
        return state.shape

    def reset(self) -> StateType:
        if self.state_transform is not None:
            return self.state_transform(self.env.reset())
        return self.env.reset()

    def render(self, mode="rgb_array"):
        if self.can_render:
            # In case of OpenAI, mode can be ['human', 'rgb_array']
            return self.env.render(mode=mode)
        else:
            print("Can't render. Sorry.")  # Yes, this is for haha

    def step(self, actions: ActionType) -> Tuple:
        """
        Each action results in a new state, reward, done flag, and info about env.
        """
        if self.is_discrete:
            actions = int(actions)
        state, reward, done, info = self.env.step(actions)
        if self.state_transform is not None:
            state = self.state_transform(state)
        if self.reward_transform is not None:
            reward = self.reward_transform(reward)
        return (state, reward, done, info)
