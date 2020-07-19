import abc
from typing import Any, Dict, Iterable, Tuple, Union

ActionType = Iterable
DoneType = bool
RewardType = Union[int, float]
StateType = Iterable

TaskStepType = Tuple[StateType, RewardType, DoneType, Any]


class TaskType(abc.ABC):

    name: str
    action_size: int
    state_size: int

    def step(self, action: ActionType) -> TaskStepType:
        return  # type: ignore

    def act(self):
        return  # type: ignore

    def render(self) -> None:
        pass

    def reset(self) -> StateType:
        return  # type: ignore


class AgentType(abc.ABC):

    name: str
    action_size: int
    last_loss: Union[int, float] = 0

    def act(self, state: StateType, noise: Any):
        pass

    def step(self, state: StateType, action: ActionType, reward: RewardType, next_state: StateType, done: DoneType):
        pass

    def describe_agent(self):
        pass

    def save_state(self, path: str):
        """
        Saves the whole agent state into a local file.
        """
        pass

    def load_state(self, path: str):
        """
        Reads the whole agent state from a local file.
        """
        pass


Hyperparameters = Dict[str, str]
