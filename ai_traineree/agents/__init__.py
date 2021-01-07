import random
import torch

from ai_traineree.types import AgentType


class AgentBase(AgentType):

    def __init__(self, *args, **kwargs):
        self._rng = random.Random()

    def seed(self, seed: int):
        """Sets a seed for all random number generators (RNG).

        Note that on top of local RNGs a global for the PyTorch is set.
        If this is undesiredable effect then:
        1) please additionally set `torch.manual_seed()` manually,
        2) let us know of your circumenstances.

        """
        self._rng.seed(seed)
        torch.manual_seed(seed)

        if hasattr(self, "buffer"):
            self.buffer.seed(seed)
