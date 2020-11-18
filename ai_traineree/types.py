import abc
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

ActionType = torch.Tensor
DoneType = bool
RewardType = Union[int, float]
StateType = torch.Tensor

Hyperparameters = Dict[str, str]
FeatureType = Union[Sequence[int], int]

TaskStepType = Tuple[StateType, RewardType, DoneType, Any]


class TaskType(abc.ABC):

    name: str
    action_size: int
    state_size: int
    is_discrete: bool

    def step(self, action: ActionType, **kwargs) -> TaskStepType:
        raise NotImplementedError

    def render(self, mode: Optional[str]=None) -> None:
        raise NotImplementedError

    def reset(self) -> StateType:
        raise NotImplementedError


class MultiAgentTaskType(TaskType):

    def reset(self) -> List[StateType]:
        raise NotImplementedError


class AgentType(abc.ABC):

    name: str
    state_size: Union[Sequence[int], int]
    action_size: int
    loss: Optional[Union[int, float]] = None
    actor_loss: Optional[Union[int, float]] = None
    critic_loss: Optional[Union[int, float]] = None

    def act(self, state: StateType, noise: Any):
        raise NotImplementedError

    def step(self, state: StateType, action: ActionType, reward: RewardType, next_state: StateType, done: DoneType):
        raise NotImplementedError

    def describe_agent(self) -> None:
        raise NotImplementedError

    def save_state(self, path: str):
        """Saves the whole agent state into a local file."""
        raise NotImplementedError

    def load_state(self, path: str):
        """Reads the whole agent state from a local file."""
        raise NotImplementedError


class MultiAgentType(abc.ABC):

    name: str
    state_size: Union[Sequence[int], int]
    action_size: int
    loss: Optional[Union[int, float]] = None
    actor_loss: Optional[Union[int, float]] = None
    critic_loss: Optional[Union[int, float]] = None
    agents: List[AgentType]
    agents_number: int

    def act(self, states: List[StateType], noise: Any) -> List[ActionType]:
        raise NotImplementedError

    def step(
        self, states: List[StateType], actions: List[ActionType], rewards: List[RewardType],
        next_states: List[StateType], dones: List[DoneType]
    ):
        raise NotImplementedError

    def describe_agent(self) -> None:
        raise NotImplementedError

    def save_state(self, path: str):
        """Saves the whole agent state into a local file."""
        raise NotImplementedError

    def load_state(self, path: str):
        """Reads the whole agent state from a local file."""
        raise NotImplementedError
