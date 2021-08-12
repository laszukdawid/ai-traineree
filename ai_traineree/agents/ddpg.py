import copy
import operator
from functools import cached_property, reduce
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.optim import Adam

from ai_traineree import DEVICE
from ai_traineree.agents import AgentBase
from ai_traineree.agents.agent_utils import hard_update, soft_update
from ai_traineree.buffers import ReplayBuffer
from ai_traineree.buffers.buffer_factory import BufferFactory
from ai_traineree.loggers import DataLogger
from ai_traineree.networks.bodies import ActorBody, CriticBody
from ai_traineree.noise import GaussianNoise
from ai_traineree.types import ActionType, AgentState, BufferState, DoneType, NetworkState, ObsType, RewardType
from ai_traineree.types.dataspace import DataSpace
from ai_traineree.utils import to_numbers_seq, to_tensor


class DDPGAgent(AgentBase):
    """
    Deep Deterministic Policy Gradients (DDPG).

    Instead of popular Ornstein-Uhlenbeck (OU) process for noise this agent uses Gaussian noise.

    This agent is intended for continuous tasks.
    """

    model = "DDPG"

    def __init__(
        self, obs_space: DataSpace, action_space: DataSpace, noise_scale: float=0.2, noise_sigma: float=0.1, **kwargs
    ):
        """
        Parameters:
            obs_space (DataSpace): Dataspace describing the input.
            action_space (DataSpace): Dataspace describing the output.
            noise_scale (float): Added noise amplitude. Default: 0.2.
            noise_sigma (float): Added noise variance. Default: 0.1.

        Keyword parameters:
            hidden_layers (tuple of ints): Shape of the hidden layers in fully connected network. Default: (64, 64).
            gamma (float): Discount value. Default: 0.99.
            tau (float): Soft-copy factor. Default: 0.002.
            actor_lr (float): Learning rate for the actor (policy). Default: 0.0003.
            critic_lr (float): Learning rate for the critic (value function). Default: 0.0003.
            max_grad_norm_actor (float) Maximum norm value for actor gradient. Default: 10.
            max_grad_norm_critic (float): Maximum norm value for critic gradient. Default: 10.
            batch_size (int): Number of samples used in learning. Default: 64.
            buffer_size (int): Maximum number of samples to store. Default: 1e6.
            warm_up (int): Number of samples to observe before starting any learning step. Default: 0.
            update_freq (int): Number of steps between each learning step. Default 1.
            number_updates (int): How many times to use learning step in the learning phase. Default: 1.

        """
        super().__init__(**kwargs)
        self.device = self._register_param(kwargs, "device", DEVICE)
        self.obs_space = obs_space
        self.action_space = action_space
        self._config['obs_space'] = self.obs_space
        self._config['action_space'] = self.action_space

        action_shape = action_space.to_feature()
        action_size = reduce(operator.mul, action_shape)

        # Reason sequence initiation.
        hidden_layers = to_numbers_seq(self._register_param(kwargs, 'hidden_layers', (64, 64)))
        self.actor = ActorBody(
            obs_space.shape, action_shape, hidden_layers=hidden_layers, gate_out=torch.tanh).to(self.device)
        self.critic = CriticBody(
            obs_space.shape, action_size, hidden_layers=hidden_layers).to(self.device)
        self.target_actor = ActorBody(
            obs_space.shape, action_shape, hidden_layers=hidden_layers, gate_out=torch.tanh).to(self.device)
        self.target_critic = CriticBody(
            obs_space.shape, action_size, hidden_layers=hidden_layers).to(self.device)

        # Noise sequence initiation
        self.noise = GaussianNoise(
            shape=action_shape, mu=1e-8, sigma=noise_sigma, scale=noise_scale, device=self.device)

        # Target sequence initiation
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        # Optimization sequence initiation.
        self.actor_lr = float(self._register_param(kwargs, 'actor_lr', 3e-4))
        self.critic_lr = float(self._register_param(kwargs, 'critic_lr', 3e-4))
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        self.max_grad_norm_actor = float(self._register_param(kwargs, "max_grad_norm_actor", 10.0))
        self.max_grad_norm_critic = float(self._register_param(kwargs, "max_grad_norm_critic", 10.0))

        self.gamma = float(self._register_param(kwargs, 'gamma', 0.99))
        self.tau = float(self._register_param(kwargs, 'tau', 0.02))
        self.batch_size = int(self._register_param(kwargs, 'batch_size', 64))
        self.buffer_size = int(self._register_param(kwargs, 'buffer_size', int(1e6)))
        self.buffer = ReplayBuffer(self.batch_size, self.buffer_size)

        self.warm_up = int(self._register_param(kwargs, 'warm_up', 0))
        self.update_freq = int(self._register_param(kwargs, 'update_freq', 1))
        self.number_updates = int(self._register_param(kwargs, 'number_updates', 1))

        # Breath, my child.
        self.reset_agent()
        self.iteration = 0
        self._loss_actor = 0.
        self._loss_critic = 0.

    def reset_agent(self) -> None:
        self.actor.reset_parameters()
        self.critic.reset_parameters()
        self.target_actor.reset_parameters()
        self.target_critic.reset_parameters()

    @property
    def loss(self) -> Dict[str, float]:
        return {'actor': self._loss_actor, 'critic': self._loss_critic}

    @loss.setter
    def loss(self, value):
        if isinstance(value, dict):
            self._loss_actor = value['actor']
            self._loss_critic = value['critic']
        else:
            self._loss_actor = value
            self._loss_critic = value

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) \
            and isinstance(o, type(self)) \
            and self._config == o._config \
            and self.buffer == o.buffer \
            and self.get_network_state() == o.get_network_state()

    @cached_property
    def action_min(self):
        return to_tensor(self.action_space.low)

    @cached_property
    def action_max(self):
        return to_tensor(self.action_space.high)

    @torch.no_grad()
    def act(self, obs: ObsType, noise: float=0.0) -> List[float]:
        """Acting on the observations. Returns action.

        Parameters:
            obs (array_like): current state
            eps (optional float): epsilon, for epsilon-greedy action selection. Default 0.

        Returns:
            action: (list float) Action values.
        """
        t_obs = to_tensor(obs).float().to(self.device)
        action = self.actor(t_obs)
        action += noise*self.noise.sample()
        action = torch.clamp(action, self.action_min, self.action_max)
        return action.cpu().numpy().tolist()

    def step(self, obs: ObsType, action: ActionType, reward: RewardType, next_obs: ObsType, done: DoneType) -> None:
        self.iteration += 1
        self.buffer.add(state=obs, action=action, reward=reward, next_state=next_obs, done=done)

        if self.iteration < self.warm_up:
            return

        if len(self.buffer) > self.batch_size and (self.iteration % self.update_freq) == 0:
            for _ in range(self.number_updates):
                self.learn(self.buffer.sample())

    def compute_value_loss(self, states, actions, next_states, rewards, dones):
        next_actions = self.target_actor.act(next_states)
        assert next_actions.shape == actions.shape, f"{next_actions.shape} != {actions.shape}"
        Q_target_next = self.target_critic.act(next_states, next_actions)
        Q_target = rewards + self.gamma * Q_target_next * (1 - dones)
        Q_expected = self.critic(states, actions)
        assert Q_expected.shape == Q_target.shape == Q_target_next.shape
        return mse_loss(Q_expected, Q_target)

    def compute_policy_loss(self, states) -> Tensor:
        """Compute Policy loss based on provided states.

        Loss = Mean(-Q(s, _a) ),
        where _a is actor's estimate based on state, _a = Actor(s).
        """
        pred_actions = self.actor(states)
        return -self.critic(states, pred_actions).mean()

    def learn(self, experiences) -> None:
        """Update critics and actors"""
        rewards = to_tensor(experiences['reward']).float().to(self.device).unsqueeze(1)
        dones = to_tensor(experiences['done']).type(torch.int).to(self.device).unsqueeze(1)
        states = to_tensor(experiences['state']).float().to(self.device)
        actions = to_tensor(experiences['action']).float().to(self.device).view((-1,) + self.action_space.shape)
        next_states = to_tensor(experiences['next_state']).float().to(self.device)

        assert rewards.shape == dones.shape == (self.batch_size, 1), f"R.shape={rewards.shape}, D.shap={dones.shape}"
        assert states.shape == next_states.shape == (self.batch_size,) + self.obs_space.shape, f"states.shape: {states.shape}"
        assert actions.shape == (self.batch_size,) + self.action_space.shape, f"actions.shape: {actions.shape}"  # type: ignore

        # Value (critic) optimization
        loss_critic = self.compute_value_loss(states, actions, next_states, rewards, dones)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm_critic)
        self.critic_optimizer.step()
        self._loss_critic = float(loss_critic.item())

        # Policy (actor) optimization
        loss_actor = self.compute_policy_loss(states)
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm_actor)
        self.actor_optimizer.step()
        self._loss_actor = loss_actor.item()

        # Soft update target weights
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)

    def state_dict(self) -> Dict[str, dict]:
        """Describes agent's networks.

        Returns:
            state: (dict) Provides actors and critics states.

        """
        return {
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict()
        }

    def log_metrics(self, data_logger: DataLogger, step: int, full_log: bool=False):
        data_logger.log_value("loss/actor", self._loss_actor, step)
        data_logger.log_value("loss/critic", self._loss_critic, step)

        if full_log:
            for idx, layer in enumerate(self.actor.layers):
                if hasattr(layer, "weight"):
                    data_logger.create_histogram(f"actor/layer_weights_{idx}", layer.weight, step)
                if hasattr(layer, "bias") and layer.bias is not None:
                    data_logger.create_histogram(f"actor/layer_bias_{idx}", layer.bias, step)

            for idx, layer in enumerate(self.critic.layers):
                if hasattr(layer, "weight"):
                    data_logger.create_histogram(f"critic/layer_weights_{idx}", layer.weight, step)
                if hasattr(layer, "bias") and layer.bias is not None:
                    data_logger.create_histogram(f"critic/layer_bias_{idx}", layer.bias, step)

    def get_state(self) -> AgentState:
        return AgentState(
            model=self.model,
            obs_space=self.obs_space,
            action_space=self.action_space,
            config=self._config,
            buffer=copy.deepcopy(self.buffer.get_state()),
            network=copy.deepcopy(self.get_network_state()),
        )

    def get_network_state(self) -> NetworkState:
        net = dict(
            actor=self.actor.state_dict(),
            target_actor=self.target_actor.state_dict(),
            critic=self.critic.state_dict(),
            target_critic=self.target_critic.state_dict(),
        )
        return NetworkState(net=net)

    @staticmethod
    def from_state(state: AgentState) -> AgentBase:
        config = copy.copy(state.config)
        config.update({'obs_space': state.obs_space, 'action_space': state.action_space})
        agent = DDPGAgent(**config)
        if state.network is not None:
            agent.set_network(state.network)
        if state.buffer is not None:
            agent.set_buffer(state.buffer)
        return agent

    def set_buffer(self, buffer_state: BufferState) -> None:
        self.buffer = BufferFactory.from_state(buffer_state)

    def set_network(self, network_state: NetworkState) -> None:
        self.actor.load_state_dict(copy.deepcopy(network_state.net['actor']))
        self.target_actor.load_state_dict(network_state.net['target_actor'])
        self.critic.load_state_dict(network_state.net['critic'])
        self.target_critic.load_state_dict(network_state.net['target_critic'])

    def save_state(self, path: str) -> None:
        agent_state = self.get_state()
        torch.save(agent_state, path)

    def load_state(self, *, path: Optional[str]=None, agent_state: Optional[dict]=None):
        if path is None and agent_state:
            raise ValueError("Either `path` or `agent_state` must be provided to load agent's state.")
        if path is not None and agent_state is None:
            agent_state = torch.load(path)
        self._config = agent_state.get('config', {})
        self.__dict__.update(**self._config)

        self.actor.load_state_dict(agent_state['actor'])
        self.critic.load_state_dict(agent_state['critic'])
        self.target_actor.load_state_dict(agent_state['target_actor'])
        self.target_critic.load_state_dict(agent_state['target_critic'])
