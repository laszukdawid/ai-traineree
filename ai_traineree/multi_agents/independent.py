from typing import Dict, List, Optional

import torch

from ai_traineree.loggers import DataLogger
from ai_traineree.types.agent import AgentType, MultiAgentType
from ai_traineree.types.experience import Experience


class IndependentAgents(MultiAgentType):

    model = "IMA"

    def __init__(self, agents: List[AgentType], agent_names: Optional[List[str]] = None, **kwargs):
        """Independent agents.

        An abstraction to manage multiple agents. It assumes no interaction between agents.

        Parameters:
            agents (list of agents): List of initiated agents. If `agent_names` aren't provided,
                the order of the list is assumed to be the same as the execution order by environment.
            agent_names (optional list of str): String names for agents passed in `agents`. These name
                should be associated with agent names expected by the environment. Default: None.

        """
        if agent_names is None:
            agent_names = [f"{agent.model}_{idx}" for (idx, agent) in enumerate(agents)]

        assert len(agent_names) == len(agents), "Expecting `agents` and `agent_names` to have the same lengths"

        self.num_agents = len(agents)
        self.agents: Dict[str, AgentType] = {agent_name: agent for (agent_name, agent) in zip(agent_names, agents)}

        self.reset()

    @property
    def loss(self) -> Dict[str, float]:
        out = {}
        for agent_name, agent in self.agents.items():
            for loss_name, loss_value in agent.loss.items():
                out[f"{agent_name}_{loss_name}"] = loss_value
        return out

    @loss.setter
    def loss(self, value):
        for agent in self.agents.values():
            agent.loss = value

    def seed(self, seed: int):
        for agent in self.agents.values():
            agent.seed(seed)

    def reset(self) -> None:
        """Resets all agents' states."""
        for agent in self.agents.values():
            agent.reset()

    def step(self, agent_name: str, experience: Experience) -> None:
        return self.agents[agent_name].step(experience)

    @torch.no_grad()
    def act(self, agent_name: str, experience: Experience, noise: float = 0.0) -> Experience:
        return self.agents[agent_name].act(experience, noise)

    def commit(self) -> None:
        """This method does nothing.

        Since all agents are completely independent there is no need for synchronizing them.
        """
        pass

    def log_metrics(self, data_logger: DataLogger, step: int, full_log: bool = False):
        for agent_name, agent in self.agents.items():
            data_logger.log_values_dict(f"{agent_name}/loss", agent.loss, step)

    def get_state(self):
        agents_state = {}
        agents_state["config"] = self._config
        for agent_name, agent in self.agents.items():
            agents_state[agent_name] = {"network": agent.state_dict(), "config": agent.hparams}
        return agents_state

    def save_state(self, path: str):
        agents_state = self.get_state()
        torch.save(agents_state, path)

    def load_state(self, path: str):
        all_agent_state = torch.load(path)
        self._config = all_agent_state.get("config", {})
        self.__dict__.update(**self._config)
        for agent_name, agent in self.agents.items():
            agent_state = all_agent_state[agent_name]
            agent.load_state(agent_state=agent_state["network"])
            agent._config = agent_state.get("config", {})
            agent.__dict__.update(**agent._config)

    def state_dict(self) -> Dict[str, dict]:
        return {name: agent.state_dict() for (name, agent) in self.agents.items()}
