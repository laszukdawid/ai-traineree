from typing import Dict, List, Sequence, Union

Numeric = Union[int, float]
ObsType = ObservationType = Union[List[int], List[float]]
StateType = Union[int, List[float]]
ActionType = Union[int, float, List]
DoneType = bool
RewardType = Union[int, float]

HyperparameterType = Dict[str, str]
FeatureType = Sequence[int]
