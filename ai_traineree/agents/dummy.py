
import copy
import numpy as np

from ai_traineree.agents import AgentBase
from ai_traineree.types import AgentState, BufferState, DataSpace, NetworkState
from typing import Callable, Dict, Optional, Type
from ai_traineree.loggers import DataLogger
from ai_traineree.experience import Experience

import warnings

class DummyAgent(AgentBase):
    """Deep Q-Learning Network (DQN).
     
     Agent that returns random values regardless of the input
     Print mean of the taken actions, observables and reward
     
     """
    model = "Dummy"
    def __init__(        self,
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
    def loss(self) -> Dict[str, float]:
        return {"loss": self._loss}

    @loss.setter
    def loss(self, value):
        if isinstance(value, dict):
            value = value["loss"]
        self._loss = value


    def reset(self):
        warnings.warn("Dummy Agent is for debugging purposes only. Nothing to reset.")
        return


    def step(self, exp: Experience) -> None:
        """Letting the agent to take a step. In this case, since the actions are totally random we don't need to save any observation

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
        s = 1
        for i in self.action_space.shape: s*=i
        action = np.random.randint(low = self.action_space.low, high = self.action_space.high+1, size = s).reshape(self.action_space.shape)
        return experience.update(action=action)

    def learn(self, experiences: Dict[str, list]) -> None:
        """No learning.

        Parameters:
            experiences: Samples experiences from the experience buffer.

        """
        return 



    def state_dict(self) -> Dict[str, dict]:
        """Describes agent's networks.

        Returns:
            state: (dict) Provides actors and critics states.

        """
        return {
            "net": "No network",
            "target_net": "No learning process",
        }

    def log_metrics(self, data_logger: DataLogger, step: int, full_log: bool = False):
        """Uses provided DataLogger to provide agent's metrics.

        Parameters:
            data_logger (DataLogger): Instance of the SummaryView, e.g. torch.utils.tensorboard.SummaryWritter.
            step (int): Ordering value, e.g. episode number.
            full_log (bool): Whether to all available information. Useful to log with lesser frequency.
        """
        data_logger.log_value("loss/agent", self._loss, step)

    def get_state(self) -> AgentState:
        """Provides agent's internal state."""
        raise Exception("Dummy Agent has no network available. This agent is just for debugging."\
            " Please refer to the ai-traineree documentation to explore other agents types.")

    def get_network_state(self) -> NetworkState:
        raise Exception("Dummy Agent has no network available. This agent is just for debugging."\
            " Please refer to the ai-traineree documentation to explore other agents types.")

    @staticmethod
    def from_state(state: AgentState) -> AgentBase:
        raise Exception("Dummy Agent has no network available. This agent is just for debugging."\
            " Please refer to the ai-traineree documentation to explore other agents types.")

    def set_buffer(self, buffer_state: BufferState) -> None:
        raise Exception("Dummy Agent has no network available. This agent is just for debugging."\
            " Please refer to the ai-traineree documentation to explore other agents types.")

    def set_network(self, network_state: NetworkState) -> None:
        raise Exception("Dummy Agent has no network available. This agent is just for debugging."\
            " Please refer to the ai-traineree documentation to explore other agents types.")

    def save_state(self, path: str):
        """Saves agent's state into a file.

        Parameters:
            path: String path where to write the state.

        """
        raise Exception("Dummy Agent has no network available. This agent is just for debugging."\
            " Please refer to the ai-traineree documentation to explore other agents types.")

    def load_state(self, *, path: Optional[str] = None, state: Optional[AgentState] = None) -> None:
        """Loads state from a file under provided path.

        Parameters:
            path: String path indicating where the state is stored.

        """
        raise Exception("Dummy Agent has no network available. This agent is just for debugging."\
            " Please refer to the ai-traineree documentation to explore other agents types.")


    def save_buffer(self, path: str) -> None:
        """Saves data from the buffer into a file under provided path.

        Parameters:
            path: String path where to write the buffer.

        """
        raise Exception("Dummy Agent has no network available. This agent is just for debugging."\
            " Please refer to the ai-traineree documentation to explore other agents types.")

    def load_buffer(self, path: str) -> None:
        """Loads data into the buffer from provided file path.

        Parameters:
            path: String path indicating where the buffer is stored.

        """
        raise Exception("Dummy Agent has no network available. This agent is just for debugging."\
            " Please refer to the ai-traineree documentation to explore other agents types.")



from ai_traineree.runners.env_runner import EnvRunner
from ai_traineree.tasks import GymTask

task = GymTask('CartPole-v1')
agent = DummyAgent(task.obs_space, task.action_space, n_steps=5)

env_runner = EnvRunner(task, agent)

# Learning
scores = env_runner.run(reward_goal=100, max_episodes=100, force_new=True)

# Check what we have learned by rendering
env_runner.interact_episode(render=True)