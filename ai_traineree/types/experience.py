from typing import Any

import jsons

from ai_traineree.utils import to_list

from .primitive import ActionType, DoneType, ObsType, RewardType


class Experience:
    """
    Basic data unit to hold information.

    It typically represents a one whole cycle of observation - action - reward.
    Data type used to store experiences in experience buffers.
    """

    whitelist = [
        "action",
        "reward",
        "done",
        "obs",
        "next_obs",
        "state",
        "next_state",
        "advantage",
        "logprob",
        "value",
        "noise",
        "noise_params",
        "priority",
        "index",
        "weight",
        "state_idx",
        "next_state_idx",
    ]
    obs: ObsType
    action: ActionType
    reward: RewardType
    done: DoneType
    next_obs: ObsType | None
    state: ObsType
    next_state: ObsType

    def __init__(self, **kwargs):
        self.data = {}
        self.update(**kwargs)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Experience) and self.data == o.data

    def get(self, key: str):
        return self.data.get(key)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in Experience.whitelist:
                self.data[key] = value
                self.__dict__[key] = value  # TODO: Delete after checking that everything is updated
        return self

    def __add__(self, o_exp):
        return self.update(**o_exp.get_dict())

    def get_dict(self, serialize=False) -> dict[str, Any]:
        if serialize:
            return {k: to_list(v) for (k, v) in self.data.items()}
        return self.data


def exprience_serialization(obj: Experience, **kwargs) -> dict[str, Any]:
    # return {k: to_list(v) for (k, v) in obj.data.items() if v is not None}
    return {k: jsons.dumps(v) for (k, v) in obj.data.items()}


jsons.set_serializer(exprience_serialization, Experience)
