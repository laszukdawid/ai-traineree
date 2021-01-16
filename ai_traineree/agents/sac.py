import numpy as np
import torch
import torch.nn as nn

from ai_traineree import DEVICE
from ai_traineree.agents import AgentBase
from ai_traineree.agents.utils import hard_update, soft_update
from ai_traineree.buffers import PERBuffer
from ai_traineree.loggers import DataLogger
from ai_traineree.networks.bodies import ActorBody, CriticBody
from ai_traineree.networks.heads import DoubleCritic
from ai_traineree.policies import MultivariateGaussianPolicy
from ai_traineree.types import FeatureType
from ai_traineree.utils import to_tensor
from torch import optim, Tensor
from typing import List, Sequence, Tuple


class SACAgent(AgentBase):
    """
    Soft Actor-Critic.

    Uses stochastic policy and dual value network (two critics).

    Based on
    "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    by Haarnoja et al. (2018) (http://arxiv.org/abs/1801.01290).
    """

    name = "SAC"

    def __init__(
        self, state_size: FeatureType, action_size: FeatureType,
        actor_lr: float=2e-3, critic_lr: float=2e-3, action_clip: Tuple[float, float]=(-1, 1),
        alpha: float=0.2, **kwargs
    ):
        """
        Parameters:
            hidden_layers: (default: (128, 128)) Shape of the hidden layers that are fully connected networks.
            gamma: (default: 0.99) Discount value.
            tau: (default: 0.02) Soft copy fraction.
            batch_size: (default 64) Number of samples in a batch.
            buffer_size: (default: 1e6) Size of the prioritized experience replay buffer.
            warm_up: (default: 0) Number of samples that needs to be observed before starting to learn.
            update_freq: (default: 1) Number of samples between policy updates.
            number_updates: (default: 1) Number of times of batch sampling/training per `update_freq`.
            alpha: (default: 0.2) Weight of log probs in value function.
            alpha_lr: (default: None) If provided, it will add alpha as a training parameters and `alpha_lr` is its learning rate.
            action_scale: (default: 1.) Scale for returned action values.
            max_grad_norm_alpha: (default: 1.) Gradient clipping for the alpha.
            max_grad_norm_actor: (default 20.) Gradient clipping for the actor.
            max_grad_norm_critic: (default: 20.) Gradient clipping for the critic.
            device: Defaults to CUDA if available.

        """
        super().__init__(**kwargs)
        self.device = kwargs.get("device", DEVICE)
        self.state_size = (state_size,) if isinstance(state_size, int) else state_size
        self.action_size = (action_size,) if isinstance(action_size, int) else action_size

        self.gamma: float = float(self._register_param(kwargs, 'gamma', 0.99))
        self.tau: float = float(self._register_param(kwargs, 'tau', 0.02))
        self.batch_size: int = int(self._register_param(kwargs, 'batch_size', 64))
        self.buffer_size: int = int(self._register_param(kwargs, 'buffer_size', int(1e6)))
        self.memory = PERBuffer(self.batch_size, self.buffer_size)

        self.warm_up: int = int(self._register_param(kwargs, 'warm_up', 0))
        self.update_freq: int = int(self._register_param(kwargs, 'update_freq', 1))
        self.number_updates: int = int(self._register_param(kwargs, 'number_updates', 1))

        # Reason sequence initiation.
        hidden_layers = kwargs.get('hidden_layers', (128, 128))
        self.policy = MultivariateGaussianPolicy(self.action_size[0], device=self.device)
        self.actor = ActorBody(self.state_size, self.policy.param_dim*self.action_size[0], hidden_layers=hidden_layers, device=self.device)

        self.double_critic = DoubleCritic(self.state_size, self.action_size[0], CriticBody, hidden_layers=hidden_layers, device=self.device)
        self.target_double_critic = DoubleCritic(self.state_size, self.action_size[0], CriticBody, hidden_layers=hidden_layers, device=self.device)

        # Target sequence initiation
        hard_update(self.target_double_critic, self.double_critic)

        # Optimization sequence initiation.
        self.target_entropy = -self.action_size[0]
        self.alpha_lr = self._register_param(kwargs, "alpha_lr")
        alpha_init = float(self._register_param(kwargs, "alpha", alpha))
        self.log_alpha = torch.tensor(np.log(alpha_init), device=self.device, requires_grad=True)

        self.actor_params = list(self.actor.parameters()) 
        self.critic_params = list(self.double_critic.parameters())
        self.actor_optimizer = optim.Adam(self.actor_params, lr=actor_lr)
        self.critic_optimizer = optim.Adam(list(self.critic_params), lr=critic_lr)
        if self.alpha_lr is not None:
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
        self.action_min = action_clip[0]
        self.action_max = action_clip[1]
        self.action_scale = self._register_param(kwargs, 'action_scale', 1)
        self.max_grad_norm_alpha: float = float(self._register_param(kwargs, "max_grad_norm_alpha", 1.0))
        self.max_grad_norm_actor: float = float(self._register_param(kwargs, "max_grad_norm_actor", 20.0))
        self.max_grad_norm_critic: float = float(self._register_param(kwargs, "max_grad_norm_critic", 20.0))

        # Breath, my child.
        self.reset_agent()
        self.iteration = 0

        self._loss_actor = float('inf')
        self._loss_critic = float('inf')

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def loss(self):
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
        self.double_critic.reset_parameters()
        self.target_double_critic.reset_parameters()

    def describe_agent(self) -> Sequence[dict]:
        """
        Returns network's weights in order:
        Actor, TargetActor, Critic, TargetCritic
        """
        return (self.actor.state_dict(), self.double_critic.state_dict(), self.target_double_critic.state_dict())

    def act(self, state, epsilon: float=0.0, deterministic=False) -> List[float]:
        if self._rng.random() < epsilon:
            random_action = torch.rand(self.action_size) * (self.action_max + self.action_min) - self.action_min
            return random_action.cpu().tolist()

        state = to_tensor(state).view(1, -1).float().to(self.device)
        action = self.actor.act(state)

        if not deterministic:
            action = self.policy(action).sample()

        action = torch.clamp(action*self.action_scale, self.action_min, self.action_max)
        return action.flatten().tolist()

    def step(self, state, action, reward, next_state, done):
        self.iteration += 1
        self.memory.add(
            state=state, action=action, reward=reward, next_state=next_state, done=done,
        )

        if self.iteration < self.warm_up:
            return

        if len(self.memory) > self.batch_size and (self.iteration % self.update_freq) == 0:
            for _ in range(self.number_updates):
                self.learn(self.memory.sample())

    def _update_value_function(self, states, actions, rewards, next_states, dones) -> Tensor:
        # critic loss
        action_mu = self.actor(next_states)
        dist = self.policy(action_mu)
        next_actions = dist.rsample()
        log_prob = self.policy.log_prob(dist, next_actions).unsqueeze(1)

        with torch.no_grad():
            Q_target_next, Q2_target_next = self.double_critic.act(next_states, next_actions)
            V_target = torch.min(Q_target_next, Q2_target_next) - self.alpha * log_prob
            Q_target = rewards + self.gamma * V_target * (1 - dones)
            Q_target = Q_target.type(torch.float32)

        Q_expected, Q2_expected = self.double_critic(states, actions)
        Q1_diff = Q_expected - Q_target
        error_1 = Q1_diff*Q1_diff
        mse_loss_1 = error_1.mean()

        Q2_diff = Q2_expected - Q_target
        error_2 = Q2_diff*Q2_diff
        mse_loss_2 = error_2.mean()

        error = error_1 + error_2
        loss_critic = mse_loss_1 + mse_loss_2

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm_critic)
        self.critic_optimizer.step()
        self._loss_critic = loss_critic.item()
        return error

    def _update_policy(self, states):
        # Compute actor loss
        actions_mu = self.actor(states)
        dist = self.policy(actions_mu)
        pred_actions = dist.rsample()
        log_prob = self.policy.log_prob(dist, pred_actions).unsqueeze(1)

        Q_actor = torch.min(*self.double_critic(states, pred_actions))
        loss_actor = (self.alpha * log_prob - Q_actor).mean()

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm_actor)
        self.actor_optimizer.step()
        self._loss_actor = loss_actor.item()

        # Update alpha
        if self.alpha_lr is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            loss_alpha.backward()
            nn.utils.clip_grad_norm_(self.log_alpha, self.max_grad_norm_alpha)
            self.alpha_optimizer.step()

    def learn(self, samples):
        """update the critics and actors of all the agents """

        rewards = to_tensor(samples['reward']).to(self.device).unsqueeze(1)
        dones = to_tensor(samples['done']).int().to(self.device).unsqueeze(1)
        states = to_tensor(samples['state']).float().to(self.device)
        next_states = to_tensor(samples['next_state']).float().to(self.device)
        actions = to_tensor(samples['action']).to(self.device)

        error = self._update_value_function(states, actions, rewards, next_states, dones)
        self._update_policy(states)

        if hasattr(self.memory, 'priority_update'):
            assert any(~torch.isnan(error))
            self.memory.priority_update(samples['index'], error.abs())

        soft_update(self.target_double_critic, self.double_critic, self.tau)

    def log_metrics(self, data_logger: DataLogger, step: int):
        data_logger.log_value("loss/actor", self._loss_actor, step)
        data_logger.log_value("loss/critic", self._loss_critic, step)
        data_logger.log_value("loss/alpha", self.alpha, step)

    def save_state(self, path: str):
        agent_state = dict(
            actor=self.actor.state_dict(),
            double_critic=self.double_critic.state_dict(),
            target_double_critic=self.target_double_critic.state_dict(),
            config=self._config,
        )
        torch.save(agent_state, path)

    def load_state(self, path: str):
        agent_state = torch.load(path)
        self._config = agent_state.get('config', {})
        self.__dict__.update(**self._config)

        self.actor.load_state_dict(agent_state['actor'])
        self.double_critic.load_state_dict(agent_state['double_critic'])
        self.target_double_critic.load_state_dict(agent_state['target_double_critic'])
