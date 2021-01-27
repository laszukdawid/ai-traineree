import abc
import torch

from ai_traineree.loggers import DataLogger
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


class AgentType(abc.ABC):

    name: str
    in_features: Tuple[int]
    state_size: int
    action_size: int
    loss: Dict[str, float]
    _config: Dict = {}

    @property
    def hparams(self):
        def make_strings_out_of_things_that_are_not_obvious_numbers(v):
            return str(v) if not isinstance(v, (int, float)) else v
        return {k: make_strings_out_of_things_that_are_not_obvious_numbers(v) for (k, v) in self._config.items()}

    def _register_param(self, source: Dict[str, Any], name: str, default_value=None, drop=False) -> Any:
        self._config[name] = value = source.get(name, default_value)
        if drop and name in source:
            del source[name]
        return value

    @abc.abstractmethod
    def act(self, state: StateType, noise: Any):
        pass

    @abc.abstractmethod
    def step(self, state: StateType, action: ActionType, reward: RewardType, next_state: StateType, done: DoneType):
        pass

    @abc.abstractmethod
    def log_metrics(self, data_logger: DataLogger, step: int, full_log: bool=False):
        pass

    @abc.abstractmethod
    def save_state(self, path: str):
        """Saves the whole agent state into a local file."""
        pass

    @abc.abstractmethod
    def load_state(self, path: str):
        """Reads the whole agent state from a local file."""
        pass


class MultiAgentType(abc.ABC):

    name: str
    in_features: Tuple[int]
    state_size: int
    action_size: int
    loss: Dict[str, float]
    agents: List[AgentType]
    agent_names: List[str]
    num_agents: int
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

    @abc.abstractmethod
    def step(
        self, states: List[StateType], actions: List[ActionType], rewards: List[RewardType],
        next_states: List[StateType], dones: List[DoneType]
    ):
        pass

    @abc.abstractmethod
    def describe_agent(self) -> Dict[str, Any]:
        """Returns description of all agent's components."""
        pass

    @abc.abstractmethod
    def save_state(self, path: str):
        """Saves the whole agent state into a local file."""
        pass

    @abc.abstractmethod
    def load_state(self, path: str):
        """Reads the whole agent state from a local file."""
        pass

    @abc.abstractmethod
    def seed(self, seed: int):
        pass
