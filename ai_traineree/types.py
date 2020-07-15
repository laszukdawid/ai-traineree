import abc
from ai_traineree import ActionType

class TaskType(abc.ABC):

    name: str
    action_size: int
    state_size: int

    def step(self, action: ActionType):
        pass

    def act(self):
        pass

    def render(self):
        pass

class AgentType(abc.ABC):

    name: str
    action_size: int

    def describe_agent(self):
        pass