import abc
from typing import Any, List, Optional, Tuple

from .dataspace import DataSpace
from .primitive import ActionType, DoneType, RewardType, StateType

TaskStepType = Tuple[StateType, RewardType, DoneType, Any]


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
    def render(self, mode: Optional[str] = None) -> None:
        pass

    @abc.abstractmethod
    def reset(self) -> StateType:
        pass


class MultiAgentTaskType(TaskType):
    @abc.abstractmethod
    def reset(self) -> List[StateType]:
        pass
