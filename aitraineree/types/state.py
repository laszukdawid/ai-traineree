from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from ai_traineree.types.dataspace import DataSpace


@dataclass
class BufferState:
    type: str
    buffer_size: int
    batch_size: int
    data: list | None = field(default=None, init=False)
    extra: dict[str, Any] | None = field(default=None, init=False, repr=False)


@dataclass
class NetworkState:
    net: dict[str, Any]

    def __eq__(self, other):
        for key, value in other.net.items():
            for left, right in zip(self.net[key].values(), value.values()):
                if not torch.all(left == right):
                    return False
        return True


@dataclass
class AgentState:
    """Fully identifies an agent"""

    model: str
    obs_space: DataSpace
    action_space: DataSpace
    config: dict[str, Any]
    network: Optional[NetworkState]
    buffer: Optional[BufferState]
