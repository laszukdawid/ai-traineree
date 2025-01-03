from typing import Sequence

Numeric = int | float
ObsType = ObservationType = list[int] | list[float]
StateType = int | list[float]
ActionType = int | float | list
DoneType = bool
RewardType = int | float

HyperparameterType = dict[str, str]
FeatureType = Sequence[int]
