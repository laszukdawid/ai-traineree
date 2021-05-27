import itertools
import operator
from functools import reduce
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from ai_traineree import DEVICE
from ai_traineree.agents import AgentBase
from ai_traineree.agents.agent_utils import hard_update, soft_update
from ai_traineree.buffers import PERBuffer
from ai_traineree.loggers import DataLogger
from ai_traineree.networks.bodies import ActorBody, CriticBody
from ai_traineree.networks.heads import DoubleCritic
from ai_traineree.policies import GaussianPolicy, MultivariateGaussianPolicySimple
from ai_traineree.types import FeatureType
from ai_traineree.utils import to_numbers_seq, to_tensor
from torch import Tensor, optim


class SACAgent(AgentBase):
    """
    Soft Actor-Critic.

    Uses stochastic policy and dual value network (two critics).

    Based on
    "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    by Haarnoja et al. (2018) (http://arxiv.org/abs/1801.01290).
    """

    name = "SAC"

    def __init__(self, in_features: FeatureType, action_size: int, **kwargs):
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
            max_grad_norm_actor: (default 10.) Gradient clipping for the actor.
            max_grad_norm_critic: (default: 10.) Gradient clipping for the critic.
            device: Defaults to CUDA if available.

        """
        super().__init__(**kwargs)
        self.device = kwargs.get("device", DEVICE)
        self.in_features: Tuple[int] = (in_features,) if isinstance(in_features, int) else tuple(in_features)
        self.state_size: int = in_features if isinstance(in_features, int) else reduce(operator.mul, in_features)
        self.action_size = action_size

        self.gamma: float = float(self._register_param(kwargs, 'gamma', 0.99))
        self.tau: float = float(self._register_param(kwargs, 'tau', 0.02))
        self.batch_size: int = int(self._register_param(kwargs, 'batch_size', 64))
        self.buffer_size: int = int(self._register_param(kwargs, 'buffer_size', int(1e6)))
        self.memory = PERBuffer(self.batch_size, self.buffer_size)

        self.action_min = self._register_param(kwargs, 'action_min', -1)
        self.action_max = self._register_param(kwargs, 'action_max', 1)
        self.action_scale = self._register_param(kwargs, 'action_scale', 1)

        self.warm_up = int(self._register_param(kwargs, 'warm_up', 0))
        self.update_freq = int(self._register_param(kwargs, 'update_freq', 1))
        self.number_updates = int(self._register_param(kwargs, 'number_updates', 1))
        self.actor_number_updates = int(self._register_param(kwargs, 'actor_number_updates', 1))
        self.critic_number_updates = int(self._register_param(kwargs, 'critic_number_updates', 1))

        # Reason sequence initiation.
        hidden_layers = to_numbers_seq(self._register_param(kwargs, 'hidden_layers', (128, 128)))
        actor_hidden_layers = to_numbers_seq(self._register_param(kwargs, 'actor_hidden_layers', hidden_layers))
        critic_hidden_layers = to_numbers_seq(self._register_param(kwargs, 'critic_hidden_layers', hidden_layers))

        self.simple_policy = bool(self._register_param(kwargs, "simple_policy", False))
        if self.simple_policy:
            self.policy = MultivariateGaussianPolicySimple(self.action_size, **kwargs)
            self.actor = ActorBody(self.state_size, self.policy.param_dim*self.action_size, hidden_layers=actor_hidden_layers, device=self.device)
        else:
            self.policy = GaussianPolicy(actor_hidden_layers[-1], self.action_size, out_scale=self.action_scale, device=self.device)
            self.actor = ActorBody(self.state_size, actor_hidden_layers[-1], hidden_layers=actor_hidden_layers[:-1], device=self.device)

        self.double_critic = DoubleCritic(
            self.in_features, self.action_size, CriticBody, hidden_layers=critic_hidden_layers, device=self.device)
        self.target_double_critic = DoubleCritic(
            self.in_features, self.action_size, CriticBody, hidden_layers=critic_hidden_layers, device=self.device)

        # Target sequence initiation
        hard_update(self.target_double_critic, self.double_critic)

        # Optimization sequence initiation.
        self.target_entropy = -self.action_size
        alpha_lr = self._register_param(kwargs, "alpha_lr")
        self.alpha_lr = float(alpha_lr) if alpha_lr else None
        alpha_init = float(self._register_param(kwargs, "alpha", 0.2))
        self.log_alpha = torch.tensor(np.log(alpha_init), device=self.device, requires_grad=True)

        actor_lr = float(self._register_param(kwargs, 'actor_lr', 3e-4))
        critic_lr = float(self._register_param(kwargs, 'critic_lr', 3e-4))

        self.actor_params = list(self.actor.parameters()) + list(self.policy.parameters())
        self.critic_params = list(self.double_critic.parameters())
        self.actor_optimizer = optim.Adam(self.actor_params, lr=actor_lr)
        self.critic_optimizer = optim.Adam(list(self.critic_params), lr=critic_lr)
        if self.alpha_lr is not None:
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
        self.max_grad_norm_alpha = float(self._register_param(kwargs, "max_grad_norm_alpha", 1.0))
        self.max_grad_norm_actor = float(self._register_param(kwargs, "max_grad_norm_actor", 10.0))
        self.max_grad_norm_critic = float(self._register_param(kwargs, "max_grad_norm_critic", 10.0))

        # Breath, my child.
        self.iteration = 0

        self._loss_actor = float('inf')
        self._loss_critic = float('inf')
        self._metrics: Dict[str, Union[float, Dict[str, float]]] = {}

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
        self.policy.reset_parameters()
        self.double_critic.reset_parameters()
        hard_update(self.target_double_critic, self.double_critic)

    def state_dict(self) -> Dict[str, dict]:
        """
        Returns network's weights in order:
        Actor, TargetActor, Critic, TargetCritic
        """
        return {
            "actor": self.actor.state_dict(),
            "policy": self.policy.state_dict(),
            "double_critic": self.double_critic.state_dict(),
            "target_double_critic": self.target_double_critic.state_dict(),
        }

    @torch.no_grad()
    def act(self, state, epsilon: float=0.0, deterministic=False) -> List[float]:
        if self.iteration < self.warm_up or self._rng.random() < epsilon:
            random_action = torch.rand(self.action_size) * (self.action_max + self.action_min) + self.action_min
            return random_action.cpu().tolist()

        state = to_tensor(state).view(1, self.state_size).float().to(self.device)
        proto_action = self.actor(state)
        action = self.policy(proto_action, deterministic)

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

    def compute_value_loss(self, states, actions, rewards, next_states, dones) -> Tuple[Tensor, Tensor]:
        Q1_expected, Q2_expected = self.double_critic(states, actions)

        with torch.no_grad():
            proto_next_action = self.actor(states)
            next_actions = self.policy(proto_next_action)
            log_prob = self.policy.logprob
            assert next_actions.shape == (self.batch_size, self.action_size)
            assert log_prob.shape == (self.batch_size, 1)

            Q1_target_next, Q2_target_next = self.target_double_critic.act(next_states, next_actions)
            assert Q1_target_next.shape == Q2_target_next.shape == (self.batch_size, 1)

            Q_min = torch.min(Q1_target_next, Q2_target_next)
            QH_target = Q_min - self.alpha * log_prob
            assert QH_target.shape == (self.batch_size, 1)

            Q_target = rewards + self.gamma * QH_target * (1 - dones)
            assert Q_target.shape == (self.batch_size, 1)

        Q1_diff = Q1_expected - Q_target
        error_1 = Q1_diff.pow(2)
        mse_loss_1 = error_1.mean()
        self._metrics['value/critic1'] = {'mean': float(Q1_expected.mean()), 'std': float(Q1_expected.std())}
        self._metrics['value/critic1_lse'] = float(mse_loss_1.item())

        Q2_diff = Q2_expected - Q_target
        error_2 = Q2_diff.pow(2)
        mse_loss_2 = error_2.mean()
        self._metrics['value/critic2'] = {'mean': float(Q2_expected.mean()), 'std': float(Q2_expected.std())}
        self._metrics['value/critic2_lse'] = float(mse_loss_2.item())

        Q_diff = Q1_expected - Q2_expected
        self._metrics['value/Q_diff'] = {'mean': float(Q_diff.mean()), 'std': float(Q_diff.std())}

        error = torch.min(error_1, error_2)
        loss = mse_loss_1 + mse_loss_2
        return loss, error

    def compute_policy_loss(self, states):
        proto_actions = self.actor(states)
        pred_actions = self.policy(proto_actions)
        log_prob = self.policy.logprob
        assert pred_actions.shape == (self.batch_size, self.action_size)

        Q_estimate = torch.min(*self.double_critic(states, pred_actions))
        assert Q_estimate.shape == (self.batch_size, 1)

        self._metrics['policy/entropy'] = -float(log_prob.detach().mean())
        loss = (self.alpha * log_prob - Q_estimate).mean()

        # Update alpha
        if self.alpha_lr is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha = -(self.alpha * (log_prob + self.target_entropy).detach()).mean()
            loss_alpha.backward()
            nn.utils.clip_grad_norm_(self.log_alpha, self.max_grad_norm_alpha)
            self.alpha_optimizer.step()

        return loss

    def learn(self, samples):
        """update the critics and actors of all the agents """

        rewards = to_tensor(samples['reward']).float().to(self.device).view(self.batch_size, 1)
        dones = to_tensor(samples['done']).int().to(self.device).view(self.batch_size, 1)
        states = to_tensor(samples['state']).float().to(self.device).view(self.batch_size, self.state_size)
        next_states = to_tensor(samples['next_state']).float().to(self.device).view(self.batch_size, self.state_size)
        actions = to_tensor(samples['action']).to(self.device).view(self.batch_size, self.action_size)

        # Critic (value) update
        for _ in range(self.critic_number_updates):
            value_loss, error = self.compute_value_loss(states, actions, rewards, next_states, dones)
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm_critic)
            self.critic_optimizer.step()
            self._loss_critic = value_loss.item()

        # Actor (policy) update
        for _ in range(self.actor_number_updates):
            policy_loss = self.compute_policy_loss(states)
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm_actor)
            self.actor_optimizer.step()
            self._loss_actor = policy_loss.item()

        if hasattr(self.memory, 'priority_update'):
            assert any(~torch.isnan(error))
            self.memory.priority_update(samples['index'], error.abs())

        soft_update(self.target_double_critic, self.double_critic, self.tau)

    def log_metrics(self, data_logger: DataLogger, step: int, full_log: bool=False):
        data_logger.log_value("loss/actor", self._loss_actor, step)
        data_logger.log_value("loss/critic", self._loss_critic, step)
        data_logger.log_value("loss/alpha", self.alpha, step)

        if self.simple_policy:
            policy_params = {str(i): v for i, v in enumerate(itertools.chain.from_iterable(self.policy.parameters()))}
            data_logger.log_values_dict("policy/param", policy_params, step)

        for name, value in self._metrics.items():
            if isinstance(value, dict):
                data_logger.log_values_dict(name, value, step)
            else:
                data_logger.log_value(name, value, step)

        if full_log:
            # TODO: Add Policy layers
            for idx, layer in enumerate(self.actor.layers):
                if hasattr(layer, "weight"):
                    data_logger.create_histogram(f"policy/layer_weights_{idx}", layer.weight, step)
                if hasattr(layer, "bias") and layer.bias is not None:
                    data_logger.create_histogram(f"policy/layer_bias_{idx}", layer.bias, step)

            for idx, layer in enumerate(self.double_critic.critic_1.layers):
                if hasattr(layer, "weight"):
                    data_logger.create_histogram(f"critic_1/layer_{idx}", layer.weight, step)
                if hasattr(layer, "bias") and layer.bias is not None:
                    data_logger.create_histogram(f"critic_1/layer_bias_{idx}", layer.bias, step)

            for idx, layer in enumerate(self.double_critic.critic_2.layers):
                if hasattr(layer, "weight"):
                    data_logger.create_histogram(f"critic_2/layer_{idx}", layer.weight, step)
                if hasattr(layer, "bias") and layer.bias is not None:
                    data_logger.create_histogram(f"critic_2/layer_bias_{idx}", layer.bias, step)

    def get_state(self):
        return dict(
            actor=self.actor.state_dict(),
            policy=self.policy.state_dict(),
            double_critic=self.double_critic.state_dict(),
            target_double_critic=self.target_double_critic.state_dict(),
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
        self.policy.load_state_dict(agent_state['policy'])
        self.double_critic.load_state_dict(agent_state['double_critic'])
        self.target_double_critic.load_state_dict(agent_state['target_double_critic'])
