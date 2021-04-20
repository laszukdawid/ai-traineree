from ai_traineree.agents import AgentBase
from ai_traineree.agents.ddpg import DDPGAgent
from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.agents.ppo import PPOAgent
from ai_traineree.agents.rainbow import RainbowAgent
from ai_traineree.types import AgentState


class AgentFactory:

    @staticmethod
    def from_state(state: AgentState) -> AgentBase:
        norm_model = state.model.upper()
        if norm_model == DQNAgent.name.upper():
            return DQNAgent.from_state(state)
        elif norm_model == PPOAgent.name.upper():
            return PPOAgent.from_state(state)
        elif norm_model == DDPGAgent.name.upper():
            return DDPGAgent.from_state(state)
        elif norm_model == RainbowAgent.name.upper():
            return RainbowAgent.from_state(state)
        else:
            raise ValueError(f"Agent state contains unsupported model type: {state.model}")
