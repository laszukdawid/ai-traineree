# import torch
from typing import Dict, List, Sequence, Union

StateType = List[float]
ActionType = Union[int, List]
DoneType = bool
RewardType = Union[int, float]

HyperparameterType = Dict[str, str]
FeatureType = Union[Sequence[int], int]
