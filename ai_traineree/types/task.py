import abc
from typing import Any

from .dataspace import DataSpace
from .primitive import ActionType, DoneType, RewardType, StateType

TaskStepType = tuple[StateType, RewardType, DoneType, Any]


class TaskType(abc.ABC):
    """
    .. _TaskType

    """

    name: str
    action_space: DataSpace
    obs_space: DataSpace
    is_discrete: bool

    @abc.abstractmethod
    def seed(self, seed):
        pass

    @abc.abstractmethod
    def step(self, action: ActionType, **kwargs) -> TaskStepType:
        pass

    @abc.abstractmethod
    def render(self, mode: str | None = None) -> None:
        pass

    @abc.abstractmethod
    def reset(self) -> StateType:
        pass


class MultiAgentTaskType(TaskType):
    @abc.abstractmethod
    def reset(self) -> list[StateType]:
        pass
