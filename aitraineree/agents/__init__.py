import random

import torch

from ai_traineree.types import AgentType


class AgentBase(AgentType):
    def __init__(self, *args, **kwargs):
        self._config = {}
        self._rng = random.Random()
        if "seed" in kwargs:
            self.seed(kwargs.get("seed"))

    def reset(self) -> None:
        """Resets data not associated with learning."""
        pass

    def seed(self, seed) -> None:
        """Sets a seed for all random number generators (RNG).

        Note that on top of local RNGs a global for the PyTorch is set.
        If this is undesiredable effect then:
        1) please additionally set `torch.manual_seed()` manually,
        2) let us know of your circumenstances.

        Parameters:
            seed: (int) Seed value for random number generators.

        """
        if not isinstance(seed, (int, float)):
            return

        self._rng.seed(seed)
        torch.manual_seed(seed)

        if hasattr(self, "buffer"):
            self.buffer.seed(seed)
