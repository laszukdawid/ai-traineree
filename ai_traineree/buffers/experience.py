import jsons

from typing import Any, Dict
from ai_traineree import to_list


class Experience:
    """
    Data type used to store experiences in experience buffers.
    """

    common_keys = ['state', 'action', 'reward', 'next_state', 'done']
    extra_keys = ['advantage', 'logprob', 'value', 'priority', 'index', 'weight', 'state_idx', 'next_state_idx']
    whitelist = common_keys + extra_keys

    def __init__(self, **kwargs):
        self.data = {}

        for (key, value) in kwargs.items():
            if key in Experience.whitelist:
                self.data[key] = value
                self.__dict__[key] = value  # TODO: Delete after checking that everything is updated

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Experience) and self.data == o.data

    def get_dict(self, serialize=False) -> Dict[str, Any]:
        if serialize:
            return {k: to_list(v) for (k, v) in self.data.items()}
        return self.data


def exprience_serialization(obj: Experience, **kwargs) -> Dict[str, Any]:
    # return {k: to_list(v) for (k, v) in obj.data.items() if v is not None}
    return {k: jsons.dumps(v) for (k, v) in obj.data.items()}


jsons.set_serializer(exprience_serialization, Experience)
