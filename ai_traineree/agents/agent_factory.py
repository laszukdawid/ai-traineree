from ai_traineree.agents import AgentBase
from ai_traineree.agents.ddpg import DDPGAgent
from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.agents.ppo import PPOAgent
from ai_traineree.types import AgentState


class AgentFactory:

    @staticmethod
    def from_state(state: AgentState) -> AgentBase:
        if state.model == DQNAgent.name:
            return DQNAgent.from_state(state)
        elif state.model == PPOAgent.name:
            return PPOAgent.from_state(state)
        elif state.model == DDPGAgent.name:
            return DDPGAgent.from_state(state)
        else:
            raise ValueError(f"Agent state contains unsupported model type: '{state.model}'")
