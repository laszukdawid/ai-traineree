import torch

from ai_traineree import DEVICE
from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.loggers import DataLogger
from ai_traineree.types import ActionType, MultiAgentType, StateType
from typing import Dict


class IQLAgents(MultiAgentType):

    name = "IQL"

    def __init__(self, state_size: int, action_size: int, num_agents: int, **kwargs):
        """Independent Q-Learning

        A set of independent Q-Learning agents (DQN implementation) that are organized
        to work as an `Multi Agent` agent. These agents have defaults as per DQNAgent class.
        All keyword paramters are passed to each agent.

        Parameters:
            state_size (int): Dimensionality of the state.
            action_size (int): Dimensionality of the action.
            num_agents (int): Number of agents.
        
        Keyword parameters:
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

        self.state_size: int = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.agent_names = kwargs.get("agent_names", map(str, range(self.num_agents)))

        kwargs['device'] = self._register_param(kwargs, "device", DEVICE)
        kwargs['hidden_layers'] = self._register_param(kwargs, 'hidden_layers', (64, 64))
        kwargs['gamma'] = float(self._register_param(kwargs, 'gamma', 0.99))
        kwargs['tau'] = float(self._register_param(kwargs, 'tau', 0.002))
        kwargs['gradient_clip'] = self._register_param(kwargs, 'gradient_clip')
        kwargs['batch_size'] = int(self._register_param(kwargs, 'batch_size', 64))
        kwargs['buffer_size'] = int(self._register_param(kwargs, 'buffer_size', int(1e6)))
        kwargs['warm_up'] = int(self._register_param(kwargs, 'warm_up', 0))
        kwargs['update_freq'] = int(self._register_param(kwargs, 'update_freq', 1))
        kwargs['number_updates'] = int(self._register_param(kwargs, 'number_updates', 1))

        self.agents: Dict[str, DQNAgent] = {
            agent_name: DQNAgent(state_size, action_size, name=agent_name, **kwargs,
            ) for agent_name in self.agent_names
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
        if isinstance(value, dict):
            self._loss_actor = value['actor']
            self._loss_critic = value['critic']
        else:
            self._loss_actor = value
            self._loss_critic = value

    def seed(self, seed: int):
        for agent in self.agents.values():
            agent.seed(seed)

    def reset(self):
        self.reset_agents()

    def reset_agents(self):
        for agent in self.agents.values():
            agent.reset()

    def step(self, agent_name: str, state: StateType, action: ActionType, reward, next_state, done) -> None:
        return self.agents[agent_name].step(state, action, reward, next_state, done)

    @torch.no_grad()
    def act(self, agent_name: str, state: StateType, noise: float=0.0) -> ActionType:
        return self.agents[agent_name].act(state, noise)

    def log_metrics(self, data_logger: DataLogger, step: int, full_log: bool=False):
        for agent in self.agents.values():
            agent.log_metrics(data_logger, step, full_log)

    def save_state(self, path: str):
        agents_state = {}
        agents_state['config'] = self._config
        for agent_id, agent in enumerate(self.agents):
            agents_state[f'agent_{agent_id}'] = agent.describe_agent()
            agents_state[f'config_{agent_id}'] = agent.hparams
        torch.save(agents_state, path)

    def load_state(self, path: str):
        agent_state = torch.load(path)
        self._config = agent_state.get('config', {})
        self.__dict__.update(**self._config)
        for agent_id, agent in enumerate(self.agents):
            agent.actor.load_state_dict(agent_state[f'actor_{agent_id}'])
            agent._config = agent_state[f'config_{agent_id}'].get(f'config_{agent_id}', {})
            agent.__dict__.update(**agent._config)
    
    def describe_agent(self):
        return [agent.describe_agent() for agent in self.agents]
