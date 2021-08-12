from functools import cached_property
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.optim import AdamW

from ai_traineree import DEVICE
from ai_traineree.agents import AgentBase
from ai_traineree.agents.agent_utils import hard_update, soft_update
from ai_traineree.buffers import ReplayBuffer
from ai_traineree.loggers import DataLogger
from ai_traineree.networks.bodies import ActorBody, CriticBody
from ai_traineree.networks.heads import DoubleCritic
from ai_traineree.noise import OUProcess
from ai_traineree.types import ActionType, DoneType, ObsType, RewardType
from ai_traineree.types.dataspace import DataSpace
from ai_traineree.utils import to_numbers_seq, to_tensor


class TD3Agent(AgentBase):
    """
    Twin Delayed Deep Deterministic (TD3) Policy Gradient.

    In short, it's a slightly modified/improved version of the DDPG. Compared to the DDPG in this package,
    which uses Guassian noise, this TD3 uses Ornstein–Uhlenbeck process as the noise.
    """

    model = "TD3"

    def __init__(self, obs_space: DataSpace, action_space: DataSpace, noise_scale: float=0.2, noise_sigma: float=0.1, **kwargs):
        """
        Parameters:
            obs_space (DataSpace): Dataspace describing the input.
            action_space (DataSpace): Dataspace describing the output.
            noise_scale (float): Added noise amplitude. Default: 0.2.
            noise_sigma (float): Added noise variance. Default: 0.1.

        Keyword parameters:
            hidden_layers (tuple of ints): Shape of the hidden layers in fully connected network. Default: (128, 128).
            actor_lr (float): Learning rate for the actor (policy). Default: 0.003.
            critic_lr (float): Learning rate for the critic (value function). Default: 0.003.
            gamma (float): Discount value. Default: 0.99.
            tau (float): Soft-copy factor. Default: 0.02.
            actor_hidden_layers (tuple of ints): Shape of network for actor. Default: `hideen_layers`.
            critic_hidden_layers (tuple of ints): Shape of network for critic. Default: `hideen_layers`.
            max_grad_norm_actor (float) Maximum norm value for actor gradient. Default: 100.
            max_grad_norm_critic (float): Maximum norm value for critic gradient. Default: 100.
            batch_size (int): Number of samples used in learning. Default: 64.
            buffer_size (int): Maximum number of samples to store. Default: 1e6.
            warm_up (int): Number of samples to observe before starting any learning step. Default: 0.
            update_freq (int): Number of steps between each learning step. Default 1.
            number_updates (int): How many times to use learning step in the learning phase. Default: 1.

        """
        super().__init__(**kwargs)
        self.device = self._register_param(kwargs, "device", DEVICE)  # Default device is CUDA if available

        # Reason sequence initiation.
        assert len(action_space.shape) == 1, "Only 1D actions are supported"
        self.obs_space = obs_space
        self.action_space = action_space
        self._config['obs_space'] = self.obs_space
        self._config['action_space'] = self.action_space
        action_size = action_space.shape[0]

        hidden_layers = to_numbers_seq(self._register_param(kwargs, 'hidden_layers', (128, 128)))
        self.actor = ActorBody(obs_space.shape, action_space.shape, hidden_layers=hidden_layers).to(self.device)
        self.critic = DoubleCritic(obs_space.shape, action_size, CriticBody, hidden_layers=hidden_layers).to(self.device)
        self.target_actor = ActorBody(obs_space.shape, action_space.shape, hidden_layers=hidden_layers).to(self.device)
        self.target_critic = DoubleCritic(obs_space.shape, action_size, CriticBody, hidden_layers=hidden_layers).to(self.device)

        # Noise sequence initiation
        # self.noise = GaussianNoise(shape=(action_size,), mu=1e-8, sigma=noise_sigma, scale=noise_scale, device=device)
        self.noise = OUProcess(shape=action_size, scale=noise_scale, sigma=noise_sigma, device=self.device)

        # Target sequence initiation
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        # Optimization sequence initiation.
        actor_lr = float(self._register_param(kwargs, 'actor_lr', 3e-3))
        critic_lr = float(self._register_param(kwargs, 'critic_lr', 3e-3))
        self.actor_optimizer = AdamW(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = AdamW(self.critic.parameters(), lr=critic_lr)
        self.max_grad_norm_actor: float = float(kwargs.get("max_grad_norm_actor", 100))
        self.max_grad_norm_critic: float = float(kwargs.get("max_grad_norm_critic", 100))

        self.gamma = float(self._register_param(kwargs, 'gamma', 0.99))
        self.tau = float(self._register_param(kwargs, 'tau', 0.02))
        self.batch_size = int(self._register_param(kwargs, 'batch_size', 64))
        self.buffer_size = int(self._register_param(kwargs, 'buffer_size', int(1e6)))
        self.buffer = ReplayBuffer(self.batch_size, self.buffer_size)

        self.warm_up = int(self._register_param(kwargs, 'warm_up', 0))
        self.update_freq = int(self._register_param(kwargs, 'update_freq', 1))
        self.update_policy_freq = int(self._register_param(kwargs, 'update_policy_freq', 1))
        self.number_updates = int(self._register_param(kwargs, 'number_updates', 1))
        self.noise_reset_freq = int(self._register_param(kwargs, 'noise_reset_freq', 10000))

        # Breath, my child.
        self.reset_agent()
        self.iteration = 0
        self._loss_actor = float('nan')
        self._loss_critic = float('nan')

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

    def reset_agent(self) -> None:
        self.actor.reset_parameters()
        self.critic.reset_parameters()
        self.target_actor.reset_parameters()
        self.target_critic.reset_parameters()

    @cached_property
    def action_min(self):
        return to_tensor(self.action_space.low)

    @cached_property
    def action_max(self):
        return to_tensor(self.action_space.high)

    @torch.no_grad()
    def act(self, obs: ObsType, epsilon: float=0.0, training_mode: bool=True) -> List[float]:
        """
        Agent acting on observations.

        When the training_mode is True (default) a noise is added to each action.
        """
        # Epsilon greedy
        if self._rng.random() < epsilon:
            rnd = torch.rand(self.action_space.shape)
            rnd_actions = rnd*(self.action_max - self.action_min) - self.action_min
            return rnd_actions.tolist()

        t_obs = to_tensor(obs).float().to(self.device)
        action = self.actor(t_obs)
        if training_mode:
            action += self.noise.sample()
        return torch.clamp(action, self.action_min, self.action_max).tolist()

    @torch.no_grad()
    def target_act(self, obs: ObsType, noise: float=0.0):
        t_obs = to_tensor(obs).float().to(self.device)
        action = self.target_actor(t_obs) + noise*self.noise.sample()
        return torch.clamp(action, self.action_min, self.action_max).cpu().numpy().astype(np.float32)

    def step(self, obs: ObsType, action: ActionType, reward: RewardType, next_obs: ObsType, done: DoneType):
        self.iteration += 1
        self.buffer.add(state=obs, action=action, reward=reward, next_state=next_obs, done=done)

        if (self.iteration % self.noise_reset_freq) == 0:
            self.noise.reset_states()

        if self.iteration < self.warm_up:
            return

        if len(self.buffer) <= self.batch_size:
            return

        if not (self.iteration % self.update_freq) or not (self.iteration % self.update_policy_freq):
            for _ in range(self.number_updates):
                # Note: Inside this there's a delayed policy update.
                #       Every `update_policy_freq` it will learn `number_updates` times.
                self.learn(self.buffer.sample())

    def learn(self, experiences):
        """Update critics and actors"""
        rewards = to_tensor(experiences['reward']).float().to(self.device).unsqueeze(1)
        dones = to_tensor(experiences['done']).type(torch.int).to(self.device).unsqueeze(1)
        states = to_tensor(experiences['state']).float().to(self.device)
        actions = to_tensor(experiences['action']).to(self.device)
        next_states = to_tensor(experiences['next_state']).float().to(self.device)

        if (self.iteration % self.update_freq) == 0:
            self._update_value_function(states, actions, rewards, next_states, dones)

        if (self.iteration % self.update_policy_freq) == 0:
            self._update_policy(states)

            soft_update(self.target_actor, self.actor, self.tau)
            soft_update(self.target_critic, self.critic, self.tau)

    def _update_value_function(self, states, actions, rewards, next_states, dones):
        # critic loss
        next_actions = self.target_actor.act(next_states)
        Q_target_next = torch.min(*self.target_critic.act(next_states, next_actions))
        Q_target = rewards + (self.gamma * Q_target_next * (1 - dones))
        Q1_expected, Q2_expected = self.critic(states, actions)
        loss_critic = mse_loss(Q1_expected, Q_target) + mse_loss(Q2_expected, Q_target)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm_critic)
        self.critic_optimizer.step()
        self._loss_critic = float(loss_critic.item())

    def _update_policy(self, states):
        # Compute actor loss
        pred_actions = self.actor(states)
        loss_actor = -self.critic(states, pred_actions)[0].mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm_actor)
        self.actor_optimizer.step()
        self._loss_actor = loss_actor.item()

    def state_dict(self) -> Dict[str, dict]:
        """Describes agent's networks.

        Returns:
            state: (dict) Provides actors and critics states.

        """
        return {
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic()
        }

    def log_metrics(self, data_logger: DataLogger, step: int, full_log: bool=False):
        data_logger.log_value("loss/actor", self._loss_actor, step)
        data_logger.log_value("loss/critic", self._loss_critic, step)

    def get_state(self):
        return dict(
            actor=self.actor.state_dict(), target_actor=self.target_actor.state_dict(),
            critic=self.critic.state_dict(), target_critic=self.target_critic.state_dict(),
            config=self._config,
        )

    def save_state(self, path: str):
        agent_state = self.get_state()
        torch.save(agent_state, path)

    def load_state(self, path: str):
        agent_state = torch.load(path)
        self._config = agent_state.get('config', {})
        self.__dict__.update(**self._config)

        self.actor.load_state_dict(agent_state['actor'])
        self.critic.load_state_dict(agent_state['critic'])
        self.target_actor.load_state_dict(agent_state['target_actor'])
        self.target_critic.load_state_dict(agent_state['target_critic'])
