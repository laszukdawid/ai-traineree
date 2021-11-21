from typing import Dict

import torch

from ai_traineree import DEVICE
from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.loggers import DataLogger
from ai_traineree.types import DataSpace, MultiAgentType
from ai_traineree.types.experience import Experience
from ai_traineree.utils import to_numbers_seq


class IQLAgents(MultiAgentType):

    model = "IQL"

    def __init__(self, obs_space: DataSpace, action_space: DataSpace, num_agents: int, **kwargs):
        """Independent Q-Learning

        A set of independent Q-Learning agents (:py:class:`DQN <DQNAgent>` implementation) that are organized
        to work as an `Multi Agent` agent. These agents have defaults as per DQNAgent class.
        All keyword paramters are passed to each agent.

        Parameters:
            obs_space (int): Dimensionality of the state.
            action_size (int): Dimensionality of the action.
            num_agents (int): Number of agents.

        Keyword Arguments:
            hidden_layers (tuple of ints): Shape for fully connected hidden layers.
            noise_scale (float): Default: 1.0. Noise amplitude.
            noise_sigma (float): Default: 0.5. Noise variance.
            actor_lr (float): Default: 0.001. Learning rate for actor network.
            gamma (float): Default: 0.99. Discount value
            tau (float): Default: 0.02. Soft copy value.
            gradient_clip (optional float): Max norm for learning gradient. If None then no clip.
            batch_size (int): Number of samples per learning.
            buffer_size (int): Number of previous samples to remember.
            warm_up (int): Number of samples to see before start learning.
            update_freq (int): How many samples between learning sessions.
            number_updates (int): How many learning cycles per learning session.

        """

        self.obs_space = obs_space
        self.action_space = action_space
        self.num_agents = num_agents
        self.agent_names = kwargs.get("agent_names", map(str, range(self.num_agents)))

        kwargs["device"] = self._register_param(kwargs, "device", DEVICE)
        kwargs["hidden_layers"] = to_numbers_seq(self._register_param(kwargs, "hidden_layers", (64, 64)))
        kwargs["gamma"] = float(self._register_param(kwargs, "gamma", 0.99))
        kwargs["tau"] = float(self._register_param(kwargs, "tau", 0.002))
        kwargs["gradient_clip"] = self._register_param(kwargs, "gradient_clip")
        kwargs["batch_size"] = int(self._register_param(kwargs, "batch_size", 64))
        kwargs["buffer_size"] = int(self._register_param(kwargs, "buffer_size", int(1e6)))
        kwargs["warm_up"] = int(self._register_param(kwargs, "warm_up", 0))
        kwargs["update_freq"] = int(self._register_param(kwargs, "update_freq", 1))
        kwargs["number_updates"] = int(self._register_param(kwargs, "number_updates", 1))

        self.agents: Dict[str, DQNAgent] = {
            agent_name: DQNAgent(obs_space, action_space, name=agent_name, **kwargs) for agent_name in self.agent_names
        }

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
        self.reset_agents()

    def reset_agents(self):
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
