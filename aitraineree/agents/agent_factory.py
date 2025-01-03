from aitraineree.agents import AgentBase
from aitraineree.agents.ddpg import DDPGAgent
from aitraineree.agents.dqn import DQNAgent
from aitraineree.agents.ppo import PPOAgent
from aitraineree.agents.rainbow import RainbowAgent
from aitraineree.agents.sac import SACAgent
from aitraineree.types import AgentState


class AgentFactory:
    @staticmethod
    def from_state(state: AgentState) -> AgentBase:
        norm_model = state.model.upper()
        if norm_model == DQNAgent.model.upper():
            return DQNAgent.from_state(state)
        elif norm_model == PPOAgent.model.upper():
            return PPOAgent.from_state(state)
        elif norm_model == DDPGAgent.model.upper():
            return DDPGAgent.from_state(state)
        elif norm_model == RainbowAgent.model.upper():
            return RainbowAgent.from_state(state)
        elif norm_model == SACAgent.model.upper():
            return SACAgent.from_state(state)
        else:
            raise ValueError(f"Agent state contains unsupported model type: {state.model}")
