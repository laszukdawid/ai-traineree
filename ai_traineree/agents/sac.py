from ai_traineree import DEVICE
from ai_traineree.agents.utils import hard_update, soft_update
from ai_traineree.buffers import ReplayBuffer as Buffer
from ai_traineree.networks.bodies import ActorBody, CriticBody
from ai_traineree.networks.heads import DoubleCritic
from ai_traineree.policies import MultivariateGaussianPolicy
from ai_traineree.types import AgentType
from ai_traineree.utils import to_tensor

import numpy as np
import torch
from torch import optim
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from typing import Sequence, Tuple, List


class SACAgent(AgentType):
    """
    Soft Actor-Critic.

    Uses stochastic policy and dual value network (two critics).

    Based on
    "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    by Haarnoja et al. (2018) (http://arxiv.org/abs/1801.01290).
    """

    name = "SAC"

    def __init__(
        self, state_size: int, action_size: int,
        actor_lr: float=2e-3, critic_lr: float=2e-3, clip: Tuple[int, int]=(-1, 1),
        alpha: float=0.2, device=None, **kwargs
    ):
        """
        :param hidden_layers: default (128, 128)
        """
        self.device = device if device is not None else DEVICE
        self.action_size = action_size
        self.gamma: float = float(kwargs.get('gamma', 0.99))
        self.tau: float = float(kwargs.get('tau', 0.02))
        self.batch_size: int = int(kwargs.get('batch_size', 64))
        self.buffer_size: int = int(kwargs.get('buffer_size', int(1e6)))
        self.memory = Buffer(self.batch_size, self.buffer_size)

        self.warm_up: int = int(kwargs.get('warm_up', 0))
        self.update_freq: int = int(kwargs.get('update_freq', 1))
        self.number_updates: int = int(kwargs.get('number_updates', 1))

        # Reason sequence initiation.
        hidden_layers = kwargs.get('hidden_layers', (128, 128))
        self.policy = MultivariateGaussianPolicy(action_size, self.batch_size, device=self.device)
        self.actor = ActorBody(state_size, self.policy.param_dim*action_size, hidden_layers=hidden_layers).to(self.device)

        self.double_critic = DoubleCritic(state_size, action_size, CriticBody, hidden_layers=hidden_layers).to(self.device)
        self.target_double_critic = DoubleCritic(state_size, action_size, CriticBody, hidden_layers=hidden_layers).to(self.device)

        # Target sequence initiation
        hard_update(self.target_double_critic, self.double_critic)

        # Optimization sequence initiation.
        self.target_entropy = -action_size
        self.alpha_lr = kwargs.get("alpha_lr")
        alpha_init = kwargs.get("alpha", alpha)
        self.log_alpha = torch.tensor(np.log(alpha_init), device=self.device, requires_grad=True)

        self.actor_params = list(self.actor.parameters()) 
        self.critic_params = list(self.double_critic.parameters())
        self.actor_optimizer = optim.Adam(self.actor_params, lr=actor_lr)
        self.critic_optimizer = optim.Adam(list(self.critic_params), lr=critic_lr)
        if self.alpha_lr is not None:
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
        self.action_min = clip[0]
        self.action_max = clip[1]
        self.action_scale = kwargs.get('action_scale', 1)
        self.max_grad_norm_alpha: float = float(kwargs.get("max_grad_norm_alpha", 1.0))
        self.max_grad_norm_actor: float = float(kwargs.get("max_grad_norm_actor", 20.0))
        self.max_grad_norm_critic: float = float(kwargs.get("max_grad_norm_critic", 20.0))

        # Breath, my child.
        self.reset_agent()
        self.iteration = 0

        self.actor_loss = np.nan
        self.critic_loss = np.nan

    @property
    def alpha(self):
        return self.log_alpha.exp()

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
        if np.random.random() < epsilon:
            return np.clip(self.action_scale*np.random.random(size=self.action_size), self.action_min, self.action_max)

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

    def _update_value_function(self, states, actions, rewards, next_states, dones):
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
        critic_loss = mse_loss(Q_expected, Q_target) + mse_loss(Q2_expected, Q_target)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_params, self.max_grad_norm_critic)
        self.critic_optimizer.step()
        self.critic_loss = critic_loss.item()

    def _update_policy(self, states):
        # Compute actor loss
        actions_mu = self.actor(states)
        dist = self.policy(actions_mu)
        pred_actions = dist.rsample()
        log_prob = self.policy.log_prob(dist, pred_actions).unsqueeze(1)

        Q_actor = torch.min(*self.double_critic(states, pred_actions))
        actor_loss = (self.alpha * log_prob - Q_actor).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor_params, self.max_grad_norm_actor)
        self.actor_optimizer.step()
        self.actor_loss = actor_loss.item()

        # Update alpha
        if self.alpha_lr is not None:
            self.alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            clip_grad_norm_(self.log_alpha, self.max_grad_norm_alpha)
            self.alpha_optimizer.step()

    def learn(self, samples):
        """update the critics and actors of all the agents """

        rewards = to_tensor(samples['reward']).to(self.device).unsqueeze(1)
        dones = to_tensor(samples['done']).int().to(self.device).unsqueeze(1)
        states = to_tensor(samples['state']).float().to(self.device)
        next_states = to_tensor(samples['next_state']).float().to(self.device)
        actions = to_tensor(samples['action']).to(self.device)

        self._update_value_function(states, actions, rewards, next_states, dones)
        self._update_policy(states)

        soft_update(self.target_double_critic, self.double_critic, self.tau)

    def log_writer(self, writer, episode):
        writer.add_scalar("loss/actor", self.actor_loss, episode)
        writer.add_scalar("loss/critic", self.critic_loss, episode)
        writer.add_scalar("loss/alpha", self.alpha, episode)

    def save_state(self, path: str):
        agent_state = dict(
            actor=self.actor.state_dict(),
            double_critic=self.double_critic.state_dict(),
            target_double_critic=self.target_double_critic.state_dict(),
        )
        torch.save(agent_state, path)

    def load_state(self, path: str):
        agent_state = torch.load(path)
        self.actor.load_state_dict(agent_state['actor'])
        self.double_critic.load_state_dict(agent_state['double_critic'])
        self.target_double_critic.load_state_dict(agent_state['target_double_critic'])
