import abc
from typing import Any, Dict, List

from ai_traineree.loggers import DataLogger
from ai_traineree.types.experience import Experience

from .dataspace import DataSpace
from .state import AgentState


class AgentType(abc.ABC):

    model: str
    obs_space: DataSpace
    action_space: DataSpace
    loss: Dict[str, float]
    train: bool = True
    _config: Dict = {}

    @property
    def hparams(self):
        def make_strings_out_of_things_that_are_not_obvious_numbers(v):
            return str(v) if not isinstance(v, (int, float)) else v

        return {k: make_strings_out_of_things_that_are_not_obvious_numbers(v) for (k, v) in self._config.items()}

    def _register_param(self, source: Dict[str, Any], name: str, default_value=None, update=False, drop=False) -> Any:
        self._config[name] = value = source.get(name, default_value)
        if drop and name in source:
            del source[name]
        elif update:
            source[name] = value
        return value

    @abc.abstractmethod
    def act(self, experience: Experience, noise: Any = None) -> Experience:
        pass

    @abc.abstractmethod
    def step(self, experience: Experience):
        pass

    @abc.abstractmethod
    def log_metrics(self, data_logger: DataLogger, step: int, full_log: bool = False):
        pass

    @abc.abstractmethod
    def get_state(self) -> AgentState:
        """Returns agent's internal state"""
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

    model: str
    obs_space: DataSpace
    action_space: DataSpace
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

    def _register_param(self, source: Dict[str, Any], name: str, default_value=None, update=False, drop=False) -> Any:
        self._config[name] = value = source.get(name, default_value)
        if drop:
            del source[name]
        elif update:
            source[name] = value
        return value

    def act(self, agent_name: str, Experience, noise: Any) -> Experience:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, agent_name: str, experience: Experience):
        pass

    @abc.abstractmethod
    def log_metrics(self, data_logger: DataLogger, step: int, full_log: bool = False):
        pass

    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Returns description of all agent's components."""
        pass

    @abc.abstractmethod
    def get_state(self) -> AgentState:
        """Returns agents' internal states"""
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

    def commit(self):
        pass
