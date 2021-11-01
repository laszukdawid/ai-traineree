from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from ai_traineree.types.dataspace import DataSpace


@dataclass
class BufferState:
    type: str
    buffer_size: int
    batch_size: int
    data: Optional[List] = field(default=None, init=False)
    extra: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)


@dataclass
class NetworkState:
    net: Dict[str, Any]

    def __eq__(self, other):
        for (key, value) in other.net.items():
            for (l, r) in zip(self.net[key].values(), value.values()):
                if not torch.all(l == r):
                    return False
        return True


@dataclass
class AgentState:
    """Fully identifies an agent"""

    model: str
    obs_space: DataSpace
    action_space: DataSpace
    config: Dict[str, Any]
    network: Optional[NetworkState]
    buffer: Optional[BufferState]
