import gym
import logging
from typing import Optional

from ai_traineree.types import AgentType, Hyperparameters, TaskType
from ai_traineree.tasks import GymTask
from . import run_env


class SageMakerExecutor:

    _logger = logging.getLogger("SageMakerExecutor")

    def __init__(self, env_name, agent_name: str, hyperparameters: Optional[Hyperparameters] = None):
        self._logger.info("Initiating SageMakerExecutor")

        env = gym.make(env_name)
        self.task = GymTask(env, env_name)
        agent: AgentType = None
        if agent_name.upper() == "DQN":
            from ai_traineree.agents.dqn import DQNAgent
            agent = DQNAgent
        elif agent_name.upper() == "PPO":
            from ai_traineree.agents.ppo import PPOAgent
            agent = PPOAgent
        elif agent_name.upper() == "DDPG":
            from ai_traineree.agents.ddpg import DDPGAgent
            agent = DDPGAgent
        else:
            self._logger.warning("No agent provided. You're given a PPO agent.")
            from ai_traineree.agents.ppo import PPOAgent
            agent = PPOAgent

        self.score_goal = int(hyperparameters.get("score_goal", 100))
        self.max_episodes = int(hyperparameters.get("max_episodes", 1000))

        self.eps_start: float = float(hyperparameters.get('eps_start', 1.0))
        self.eps_end: float = float(hyperparameters.get('eps_end', 0.002))
        self.eps_decay: float = float(hyperparameters.get('eps_decay', 0.999))

        self.agent: AgentType = agent(self.task.state_size, self.task.action_size, config=hyperparameters)

    def run(self) -> None:
        self._logger.info("Running model '%s' for env '%s'", self.agent.name, self.task.name)
        scores = run_env(self.task, self.agent, self.score_goal, self.max_episodes,
                eps_start=self.eps_start, eps_end=self.eps_end, eps_decay=self.eps_decay)
        # interact_episode(self.task, self.agent, 0, render=True)
    
    def save_results(self, path):
        self._logger.info("Saving the model to path %s", path)
        self.agent.save_state(path)
