import abc

from typing import Any, List, Optional, Tuple

from .primitive import ActionType, DoneType, StateType, RewardType


TaskStepType = Tuple[StateType, RewardType, DoneType, Any]


class TaskType(abc.ABC):
    """
    .. _TaskType

    """

    name: str
    action_size: int
    state_size: int
    is_discrete: bool

    @abc.abstractmethod
    def seed(self, seed):
        pass

    @abc.abstractmethod
    def step(self, action: ActionType, **kwargs) -> TaskStepType:
        pass

    @abc.abstractmethod
    def render(self, mode: Optional[str]=None) -> None:
        pass

    @abc.abstractmethod
    def reset(self) -> StateType:
        pass


class MultiAgentTaskType(TaskType):

    @abc.abstractmethod
    def reset(self) -> List[StateType]:
        pass
