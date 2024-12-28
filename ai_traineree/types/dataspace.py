from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
from torch import Tensor

from ai_traineree.types.primitive import FeatureType, Numeric
from ai_traineree.utils import condens_ndarray, to_numbers_seq


@dataclass
class DataSpace:
    dtype: str
    shape: tuple[int]
    low: Numeric | Tensor | None = None
    high: Numeric | Tensor | None = None

    @staticmethod
    def from_int(size: int):
        return DataSpace(dtype="int", shape=(size,), low=0, high=(size - 1))

    @staticmethod
    def from_gym_space(space) -> DataSpace:
        if isinstance(space, gym.spaces.multi_discrete.MultiDiscrete):
            return DataSpace(
                dtype=str(space.dtype),
                shape=(len(space.nvec),),
                low=0,
                high=to_numbers_seq(space.nvec),
            )
        elif "Discret" in str(space):
            return DataSpace(
                dtype=str(space.dtype),
                shape=(1,),
                low=0,
                high=space.n - 1,  # Inclusive bounds, so n=2 -> [0,1]
            )
        else:
            low = condens_ndarray(space.low)
            if not isinstance(low, (int, float)):
                low = low.tolist()
            high = condens_ndarray(space.high)
            if not isinstance(high, (int, float)):
                high = high.tolist()
            return DataSpace(
                shape=space.shape,
                dtype=str(space.dtype),
                low=low,
                high=high,
            )

    def to_feature(self) -> FeatureType:
        """Extracts required FeatureType from DataSpace.

        The reason this method exists is because DataSpace can be both discrete and continuous.
        Even though in both cases dataspace shape represent tensor shape of the space, it is usually used
        differently depending on the problem. Methods that work with discrete spaces will need to select
        on of the discrete values meaning that they need to know the total size (high - low),
        whereas in case of continuous space the shape is enough.

        Likely this method isn't necessary with some cleaver dataspace arrangement or additional property.
        However, for now, it simplifies.
        """
        if self.dtype.startswith("int") and len(self.shape) == 1:
            return (int(self.high) - int(self.low) + 1,)
        return self.shape
