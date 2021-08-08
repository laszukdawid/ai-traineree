from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import gym
from torch import Tensor

from ai_traineree.types.primitive import Numeric
from ai_traineree.utils import to_numbers_seq


@dataclass
class DataSpace:
    dtype: str
    shape: Tuple[int]
    low: Optional[Union[Numeric, Tensor]] = None
    high: Optional[Union[Numeric, Tensor]] = None

    @staticmethod
    def from_int(size: int):
        return DataSpace(dtype="int", shape=(size, ), low=0, high=(size-1))

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
            return DataSpace(
                shape=space.shape,
                dtype=str(space.dtype),
                low=space.low.tolist(),
                high=space.high.tolist(),
            )
