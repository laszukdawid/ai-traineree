from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ai_traineree import DEVICE
from ai_traineree.agents.agent_utils import hard_update, soft_update
from ai_traineree.agents.ddpg import DDPGAgent
from ai_traineree.buffers.replay import ReplayBuffer
from ai_traineree.loggers import DataLogger
from ai_traineree.networks.bodies import CriticBody
from ai_traineree.types import MultiAgentType
from ai_traineree.types.dataspace import DataSpace
from ai_traineree.types.experience import Experience
from ai_traineree.utils import to_numbers_seq, to_tensor


class MADDPGAgent(MultiAgentType):

    model = "MADDPG"

    def __init__(self, obs_space: DataSpace, action_space: DataSpace, num_agents: int, **kwargs):
        """Initiation of the Multi Agent DDPG.

        All keywords are also passed to DDPG agents.

        Parameters:
            obs_size (int): Dimensionality of the state.
            action_size (int): Dimensionality of the action.
            num_agents (int): Number of agents.

        Keyword Arguments:
            hidden_layers (tuple of ints): Shape for fully connected hidden layers.
            noise_scale (float): Default: 1.0. Noise amplitude.
            noise_sigma (float): Default: 0.5. Noise variance.
            actor_lr (float): Default: 0.001. Learning rate for actor network.
            critic_lr (float): Default: 0.001. Learning rate for critic network.
            gamma (float): Default: 0.99. Discount value
            tau (float): Default: 0.02. Soft copy value.
            gradient_clip (optional float): Max norm for learning gradient. If None then no clip.
            batch_size (int): Number of samples per learning.
            buffer_size (int): Number of previous samples to remember.
            warm_up (int): Number of samples to see before start learning.
            update_freq (int): How many samples between learning sessions.
            number_updates (int): How many learning cycles per learning session.

        """

        self.device = self._register_param(kwargs, "device", DEVICE, update=True)
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_agents: int = num_agents
        self.agent_names: List[str] = kwargs.get("agent_names", map(str, range(self.num_agents)))

        hidden_layers = to_numbers_seq(self._register_param(kwargs, "hidden_layers", (100, 100), update=True))
        noise_scale = float(self._register_param(kwargs, "noise_scale", 0.5))
        noise_sigma = float(self._register_param(kwargs, "noise_sigma", 1.0))
        actor_lr = float(self._register_param(kwargs, "actor_lr", 3e-4))
        critic_lr = float(self._register_param(kwargs, "critic_lr", 3e-4))

        self.agents: Dict[str, DDPGAgent] = OrderedDict(
            {
                agent_name: DDPGAgent(
                    obs_space,
                    action_space,
                    actor_lr=actor_lr,
                    critic_lr=critic_lr,
                    noise_scale=noise_scale,
                    noise_sigma=noise_sigma,
                    **kwargs,
                )
                for agent_name in self.agent_names
            }
        )

        self.gamma = float(self._register_param(kwargs, "gamma", 0.99))
        self.tau = float(self._register_param(kwargs, "tau", 0.02))
        self.gradient_clip: Optional[float] = self._register_param(kwargs, "gradient_clip")

        self.batch_size = int(self._register_param(kwargs, "batch_size", 64))
        self.buffer_size = int(self._register_param(kwargs, "buffer_size", int(1e6)))
        self.buffer = ReplayBuffer(self.batch_size, self.buffer_size)

        self.warm_up = int(self._register_param(kwargs, "warm_up", 0))
        self.update_freq = int(self._register_param(kwargs, "update_freq", 1))
        self.number_updates = int(self._register_param(kwargs, "number_updates", 1))

        assert len(obs_space.shape) == 1, "Only 1D obs spaces are supported now"
        assert len(action_space.shape) == 1, "Only 1D action spaces are supported now"
        ma_obs_shape = (num_agents * obs_space.shape[0],)
        ma_action_size = num_agents * action_space.shape[0]
        self.critic = CriticBody(ma_obs_shape, ma_action_size, hidden_layers=hidden_layers).to(self.device)
        self.target_critic = CriticBody(ma_obs_shape, ma_action_size, hidden_layers=hidden_layers).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        hard_update(self.target_critic, self.critic)

        self._step_data = {}
        self._loss_critic: float = float("nan")
        self._loss_actor: Dict[str, float] = {name: float("nan") for name in self.agent_names}
        self.reset()

    @property
    def loss(self) -> Dict[str, float]:
        out = {}
        for agent_name, agent in self.agents.items():
            for loss_name, loss_value in agent.loss.items():
                out[f"{agent_name}_{loss_name}"] = loss_value
            out[f"{agent_name}_actor"] = self._loss_actor[agent_name]
        out["critic"] = self._loss_critic
        return out

    def reset(self):
        self.iteration = 0
        self.reset_agents()

    def reset_agents(self):
        for agent in self.agents.values():
            agent.reset_agent()
        self.critic.reset_parameters()
        self.target_critic.reset_parameters()

    def step(self, agent_name: str, experience: Experience) -> None:
        self._step_data[agent_name] = experience

    def commit(self):
        step_data = defaultdict(list)
        for agent in self.agents:
            agent_experience: Experience = self._step_data[agent]
            step_data["obs"].append(agent_experience.obs)
            step_data["action"].append(agent_experience.action)
            step_data["reward"].append(agent_experience.reward)
            step_data["next_obs"].append(agent_experience.next_obs)
            step_data["done"].append(agent_experience.done)

        self.buffer.add(**step_data)
        self._step_data = {}
        self.iteration += 1
        if self.iteration < self.warm_up:
            return

        if len(self.buffer) > self.batch_size and (self.iteration % self.update_freq) == 0:
            for _ in range(self.number_updates):
                samples = self.buffer.sample()
                for agent_name in self.agents:
                    self.learn(samples, agent_name)
            self.update_targets()

    @torch.no_grad()
    def act(self, agent_name: str, experience: Experience, noise: float = 0.0) -> Experience:
        """Get actions from all agents. Synchronized action.

        Parameters:
            states: List of states per agent. Positions need to be consistent.
            noise: Scale for the noise to include

        Returns:
            actions: List of actions that each agent wants to perform

        """
        experience.update(obs=experience.obs)
        agent = self.agents[agent_name]
        experience = agent.act(experience, noise)
        return experience

    def __flatten_actions(self, actions):
        return actions.view(-1, self.num_agents * self.action_space.shape[0])

    def learn(self, experiences, agent_name: str) -> None:
        """update the critics and actors of all the agents"""
        ma_obs_size = self.num_agents * self.obs_space.shape[0]
        ma_action_size = self.num_agents * self.action_space.shape[0]

        # TODO: Just look at this mess.
        agent_number = list(self.agents).index(agent_name)
        agent_rewards = to_tensor(experiences["reward"]).select(1, agent_number).unsqueeze(-1).float().to(self.device)
        agent_dones = (
            to_tensor(experiences["done"]).select(1, agent_number).unsqueeze(-1).type(torch.int).to(self.device)
        )

        obss = (
            to_tensor(experiences["obs"])
            .to(self.device)
            .view((self.batch_size, self.num_agents) + self.obs_space.shape)
        )
        actions = to_tensor(experiences["action"]).to(self.device)
        next_obss = (
            to_tensor(experiences["next_obs"])
            .float()
            .to(self.device)
            .view((self.batch_size, self.num_agents) + self.obs_space.shape)
        )

        flat_obss = obss.view(-1, ma_obs_size)
        flat_next_obss = next_obss.view(-1, ma_obs_size)
        flat_actions = actions.view(-1, ma_action_size)
        assert agent_rewards.shape == agent_dones.shape == (self.batch_size, 1)
        assert obss.shape == next_obss.shape == (self.batch_size, self.num_agents) + self.obs_space.shape
        assert actions.shape == (self.batch_size, self.num_agents) + self.action_space.shape
        assert flat_actions.shape == (self.batch_size, ma_action_size)

        agent = self.agents[agent_name]

        next_actions = actions.detach().clone()
        next_actions.data[:, agent_number] = agent.target_actor(next_obss[:, agent_number, :])
        assert next_actions.shape == (self.batch_size, self.num_agents) + self.action_space.shape

        # critic loss
        Q_target_next = self.target_critic(flat_next_obss, self.__flatten_actions(next_actions))
        Q_target = agent_rewards + (self.gamma * Q_target_next * (1 - agent_dones))
        Q_expected = self.critic(flat_obss, flat_actions)
        loss_critic = F.mse_loss(Q_expected, Q_target)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        if self.gradient_clip:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()
        self._loss_critic = float(loss_critic.mean().item())

        # Compute actor loss
        pred_actions = actions.detach().clone()
        # pred_actions.data[:, agent_number] = agent.actor(flat_obss)
        pred_actions.data[:, agent_number] = agent.actor(obss[:, agent_number, :])

        loss_actor = -self.critic(flat_obss, self.__flatten_actions(pred_actions)).mean()
        agent.actor_optimizer.zero_grad()
        loss_actor.backward()
        agent.actor_optimizer.step()
        self._loss_actor[agent_name] = loss_actor.mean().item()

    def update_targets(self):
        """soft update targets"""
        for agent in self.agents.values():
            soft_update(agent.target_actor, agent.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)

    def log_metrics(self, data_logger: DataLogger, step: int, full_log: bool = False):
        data_logger.log_value("loss/critic", self._loss_critic, step)
        for agent_name, agent in self.agents.items():
            data_logger.log_values_dict(f"{agent_name}/loss", agent.loss, step)

    def get_state(self) -> Dict[str, dict]:
        """Returns agents' internal states"""
        agents_state = {}
        agents_state["config"] = self._config
        for agent_name, agent in self.agents.items():
            agents_state[agent_name] = {"state": agent.state_dict(), "config": agent._config}
        return agents_state

    def save_state(self, path: str):
        """Saves current state of the Multi Agent instance and all related agents.

        All states are stored via PyTorch's :func:`save <torch.save>` function.

        Parameters:
            path: (str) String path to a location where the state is store.

        """
        agents_state = self.get_state()
        torch.save(agents_state, path)

    def load_state(self, *, path: Optional[str] = None, agent_state: Optional[dict] = None) -> None:
        """Loads the state into the Multi Agent.

        The state can be provided either via path to a file that contains the state,
        see :meth:`save_state <self.save_state>`, or direclty via `state`.

        Parameters:
            path: (str) A path where the state was saved via `save_state`.
            state: (dict) Already loaded state kept in memory.

        """
        if path is None and agent_state is None:
            raise ValueError("Either `path` or `agent_state` must be provided to load agent's state.")
        if path is not None and agent_state is None:
            agent_state = torch.load(path)
        assert agent_state is not None, "Can't load state if neither agent state or valid path is provided"

        self._config = agent_state.get("config", {})
        self.__dict__.update(**self._config)
        for agent_name, agent in self.agents.items():
            _agent_state = agent_state[agent_name]
            agent.load_state(agent_state=_agent_state["state"])
            agent._config = _agent_state["config"]
            agent.__dict__.update(**agent._config)

    def seed(self, seed: int) -> None:
        for agent in self.agents.values():
            agent.seed(seed)

    def state_dict(self) -> Dict[str, Any]:
        return {name: agent.state_dict() for (name, agent) in self.agents.items()}
