from ai_traineree.types import StateType, TaskType

from typing import Tuple


class GymTask(TaskType):
    def __init__(self, env, env_name: str, state_transform=None, can_render=True):
        self.name = env_name
        self.env = env
        self.can_render = can_render
        self.is_discrete = "Discrete" in str(type(env.action_space))

        self.state_size = env.observation_space.shape[0]
        self.action_size = self.__determine_action_size(env.action_space)
        self.state_transform = state_transform

    @staticmethod
    def __determine_action_size(action_space):
        if "Discrete" in str(type(action_space)):
            return action_space.n
        else:
            return sum(action_space.shape)

    def reset(self) -> StateType:
        if self.state_transform is not None:
            return self.state_transform(self.env.reset())
        return self.env.reset()

    def render(self, mode="rgb_array"):
        if self.can_render:
            # In case of OpenAI, mode can be ['human', 'rgb_array']
            return self.env.render(mode)
        else:
            print("Can't render. Sorry.")  # Yes, this is for haha

    def step(self, actions) -> Tuple:
        """
        Each action results in a new state, reward, done flag, and info about env.
        """
        if self.is_discrete:
            actions = int(actions)
        state, reward, done, info = self.env.step(actions)
        if self.state_transform is not None:
            state = self.state_transform(state)
        return (state, reward, done, info)
