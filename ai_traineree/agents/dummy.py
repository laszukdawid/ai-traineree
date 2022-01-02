import numpy as np

from ai_traineree.agents import AgentBase
from ai_traineree.types import DataSpace
from ai_traineree.types.experience import Experience


class DummyAgent(AgentBase):
    """Dummy Agent.

    Agent that returns random values in specified shapes.
    """

    model = "Dummy"

    def __init__(self, obs_space: DataSpace, action_space: DataSpace, **kwargs):
        """Initiates Dummy agent.

        Parameters:
            obs_space (DataSpace): Dataspace describing the input.
            action_space (DataSpace): Dataspace describing the output.

        """
        super().__init__(**kwargs)
        self.obs_space = obs_space
        self.action_space = action_space

        self.loss = {"loss": 0}

    @staticmethod
    def step(*args, **kwargs) -> None:
        """Simulates transition to the next internal stage."""
        pass

    def act(self, experience: Experience, *args, **kwargs) -> Experience:
        """Returns random discrete action and updates experience.

        Parameters:
            experience (Experience): Structure holding current iteration information.

        Returns:
            Experience updated in current action.

        """
        assert self.action_space.high is not None, "ActionSpace needs to have high boundary"

        action = np.random.randint(
            low=self.action_space.low, high=self.action_space.high + 1, size=self.action_space.shape
        )
        return experience.update(action=action)

    @staticmethod
    def log_metrics(*args, **kwargs) -> None:
        """Dummy method. Doesn't do anything."""
        pass

    @staticmethod
    def get_state(*args, **kwargs) -> None:
        """Dummy method. Doesn't do anything."""
        pass

    @staticmethod
    def save_state(*args, **kwargs) -> None:
        """Dummy method. Doesn't do anything."""
        pass

    @staticmethod
    def load_state(*args, **kwargs) -> None:
        """Dummy method. Doesn't do anything."""
        pass
