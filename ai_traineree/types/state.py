import torch

from dataclasses import dataclass
from typing import Any, Dict, List

from .primitive import StateType, ActionType


@dataclass
class AgentState:
    name: str
    state_space: StateType
    action_space: ActionType
    config: Dict[str, Any]


@dataclass
class BufferState:
    type: str
    data: List


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
class FullState:
    agent: AgentState
    buffer: BufferState
    network: NetworkState
