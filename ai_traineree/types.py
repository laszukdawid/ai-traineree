import abc
from abc import abstractmethod
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

    @abstractmethod
    def step(self, action: ActionType, **kwargs) -> TaskStepType:
        pass

    @abstractmethod
    def render(self, mode: Optional[str]=None) -> None:
        pass

    @abstractmethod
    def reset(self) -> StateType:
        pass


class MultiAgentTaskType(TaskType):

    @abstractmethod
    def reset(self) -> List[StateType]:
        pass


class AgentType(abc.ABC):

    name: str
    state_size: Union[Sequence[int], int]
    action_size: int
    loss: Dict[str, float]
    _config: Dict = {}

    @property
    def hparams(self):
        def make_strings_out_of_things_that_are_not_obvious_numbers(v):
            return str(v) if v is not isinstance(v, (int, float)) else v
        return {k: make_strings_out_of_things_that_are_not_obvious_numbers(v) for (k, v) in self._config.items()}

    def _register_param(self, source: Dict[str, Any], name: str, default_value=None, drop=False) -> Any:
        self._config[name] = value = source.get(name, default_value)
        if drop and name in source:
            del source[name]
        return value

    @abstractmethod
    def act(self, state: StateType, noise: Any):
        pass

    @abstractmethod
    def step(self, state: StateType, action: ActionType, reward: RewardType, next_state: StateType, done: DoneType):
        pass

    @abstractmethod
    def save_state(self, path: str):
        """Saves the whole agent state into a local file."""
        pass

    @abstractmethod
    def load_state(self, path: str):
        """Reads the whole agent state from a local file."""
        pass


class MultiAgentType(abc.ABC):

    name: str
    state_size: Union[Sequence[int], int]
    action_size: int
    loss: Dict[str, float]
    agents: List[AgentType]
    agents_number: int
    _config: Dict = {}

    @property
    def hparams(self):
        def make_strings_out_of_things_that_are_not_obvious_numbers(v):
            return str(v) if v is not isinstance(v, (int, float)) else v
        return {k: make_strings_out_of_things_that_are_not_obvious_numbers(v) for (k, v) in self._config.items()}

    def _register_param(self, source: Dict[str, Any], name: str, default_value=None, drop=False) -> Any:
        self._config[name] = value = source.get(name, default_value)
        if drop:
            del source[name]
        return value

    def act(self, states: List[StateType], noise: Any) -> List[ActionType]:
        raise NotImplementedError

    @abstractmethod
    def step(
        self, states: List[StateType], actions: List[ActionType], rewards: List[RewardType],
        next_states: List[StateType], dones: List[DoneType]
    ):
        pass

    @abstractmethod
    def describe_agent(self) -> Dict[str, Any]:
        """Returns description of all agent's components."""
        pass

    @abstractmethod
    def save_state(self, path: str):
        """Saves the whole agent state into a local file."""
        pass

    @abstractmethod
    def load_state(self, path: str):
        """Reads the whole agent state from a local file."""
        pass
