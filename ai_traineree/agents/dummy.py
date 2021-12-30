import warnings
from typing import Callable, Dict, Optional

import numpy as np

from ai_traineree.agents import AgentBase
from ai_traineree.experience import Experience
from ai_traineree.types import AgentState, BufferState, DataSpace, NetworkState


class DummyAgent(AgentBase):

    """Dummy Agent.

    Agent that returns random values in specified shapes.
    """

    model = "Dummy"

    def __init__(
        self,
        obs_space: DataSpace,
        action_space: DataSpace,
        state_transform: Optional[Callable] = None,
        reward_transform: Optional[Callable] = None,
        **kwargs
    ):

        """Initiates Dummy agent.

        Parameters:
            obs_space (DataSpace): Dataspace describing the input.
            action_space (DataSpace): Dataspace describing the output.
            state_transform (optional func): Function to transform (encode) state before used by the network.
            reward_transform (optional func): Function to transform reward before use.

        """
        super().__init__(**kwargs)
        self.obs_space = obs_space
        self.action_space = action_space
        self.iteration: int = 0

        self._loss: float = float("nan")

    @property
    def loss(self):
        return {"loss": self._loss}

    def reset(self):
        warnings.warn("Dummy Agent is for debugging purposes only. Nothing to reset.")
        return

    @staticmethod
    def step(exp: Experience) -> None:

        """Letting the agent to take a step.
        In this case, since the actions are totally random we don't need to save any observation

        Parameters:
            obs (ObservationType): Observation.
            action (int): Discrete action associated with observation.
            reward (float): Reward obtained for taking action at state.
            next_obs (ObservationType): Observation in a state where the action took.
            done: (bool) Whether in terminal (end of episode) state.

        """
        return

    def act(self, experience: Experience, eps: float = 0.0) -> Experience:

        """Returns random action.

        Parameters:
            obs (array_like): current observation
            eps (optional float): epsilon, for epsilon-greedy action selection. Default 0.

        Returns:
            Random action taken from the agent.

        """
        action_size = 1
        for i in self.action_space.shape:
            action_size *= i
        action = np.random.randint(
            low=self.action_space.low, high=self.action_space.high + 1, size=action_size
        ).reshape(self.action_space.shape)
        return experience.update(action=action)

    @staticmethod
    def learn(self, experiences: Dict[str, list]) -> None:

        """No learning.

        Parameters:
            experiences: Samples experiences from the experience buffer.

        """
        return

    @staticmethod
    def state_dict() -> Dict[str, dict]:

        """Describes agent's networks.

        Returns:
            state: (dict) Provides actors and critics states.

        """
        return {
            "net": "No network",
            "target_net": "No learning process",
        }

    def log_metrics(self, data_logger: None, step: int, full_log: bool = False):

        """Uses provided DataLogger to provide agent's metrics.

        Parameters:
            data_logger (DataLogger): Instance of the SummaryView, e.g. torch.utils.tensorboard.SummaryWritter.
            step (int): Ordering value, e.g. episode number.
            full_log (bool): Whether to all available information. Useful to log with lesser frequency.
        """
        data_logger.log_value("loss/agent", self._loss, step)

    @staticmethod
    def get_state():

        """Provides agent's internal state."""
        raise Exception(
            "Dummy Agent has no network available. This agent is just for debugging."
            " Please refer to the ai-traineree documentation to explore other agents types."
        )

    @staticmethod
    def get_network_state():
        raise Exception(
            "Dummy Agent has no network available. This agent is just for debugging."
            " Please refer to the ai-traineree documentation to explore other agents types."
        )

    @staticmethod
    def from_state(state: AgentState) -> AgentBase:
        raise Exception(
            "Dummy Agent has no network available. This agent is just for debugging."
            " Please refer to the ai-traineree documentation to explore other agents types."
        )

    @staticmethod
    def set_buffer(buffer_state: BufferState) -> None:
        raise Exception(
            "Dummy Agent has no network available. This agent is just for debugging."
            " Please refer to the ai-traineree documentation to explore other agents types."
        )

    @staticmethod
    def set_network(network_state: NetworkState) -> None:
        raise Exception(
            "Dummy Agent has no network available. This agent is just for debugging."
            " Please refer to the ai-traineree documentation to explore other agents types."
        )

    @staticmethod
    def save_state(path: str):

        """Saves agent's state into a file.

        Parameters:
            path: String path where to write the state.

        """
        raise Exception(
            "Dummy Agent has no network available. This agent is just for debugging."
            " Please refer to the ai-traineree documentation to explore other agents types."
        )

    @staticmethod
    def load_state(*, path: Optional[str] = None, state: Optional[AgentState] = None) -> None:

        """Loads state from a file under provided path.

        Parameters:
            path: String path indicating where the state is stored.

        """
        raise Exception(
            "Dummy Agent has no network available. This agent is just for debugging."
            " Please refer to the ai-traineree documentation to explore other agents types."
        )

    @staticmethod
    def save_buffer(path: str) -> None:

        """Saves data from the buffer into a file under provided path.

        Parameters:
            path: String path where to write the buffer.

        """
        raise Exception(
            "Dummy Agent has no network available. This agent is just for debugging."
            " Please refer to the ai-traineree documentation to explore other agents types."
        )

    @staticmethod
    def load_buffer(path: str) -> None:

        """Loads data into the buffer from provided file path.

        Parameters:
            path: String path indicating where the buffer is stored.

        """
        raise Exception(
            "Dummy Agent has no network available. This agent is just for debugging."
            " Please refer to the ai-traineree documentation to explore other agents types."
        )
