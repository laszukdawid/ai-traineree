import abc
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

ActionType = Iterable
DoneType = bool
RewardType = Union[int, float]
StateType = Iterable

Hyperparameters = Dict[str, str]

TaskStepType = Tuple[StateType, RewardType, DoneType, Any]


class TaskType(abc.ABC):

    name: str
    action_size: int
    state_size: int

    def step(self, action: ActionType) -> TaskStepType:
        raise NotImplementedError

    def act(self):
        raise NotImplementedError

    def render(self, mode: Optional[str]=None) -> None:
        raise NotImplementedError

    def reset(self) -> StateType:
        raise NotImplementedError


class AgentType(abc.ABC):

    name: str
    state_size: Union[Sequence[int], int]
    action_size: int
    last_loss: Union[int, float] = 0
    actor_loss: Union[int, float] = 0
    critic_loss: Union[int, float] = 0

    def act(self, state: StateType, noise: Any):
        raise NotImplementedError

    def step(self, state: StateType, action: ActionType, reward: RewardType, next_state: StateType, done: DoneType):
        raise NotImplementedError

    def describe_agent(self):
        raise NotImplementedError

    def save_state(self, path: str):
        """
        Saves the whole agent state into a local file.
        """
        raise NotImplementedError

    def load_state(self, path: str):
        """
        Reads the whole agent state from a local file.
        """
        raise NotImplementedError
