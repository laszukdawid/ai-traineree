import torch

from ai_traineree import DEVICE
from ai_traineree.agents.utils import hard_update, soft_update
from ai_traineree.buffers import ReplayBuffer
from ai_traineree.networks.bodies import ActorBody, CriticBody
from ai_traineree.noise import GaussianNoise
from ai_traineree.types import AgentType
from ai_traineree.utils import to_tensor
from numpy import float32 as _float32
from torch.optim import Adam
from torch.nn.functional import mse_loss
from typing import Any, Sequence, Tuple


class DDPGAgent(AgentType):
    """
    Deep Deterministic Policy Gradients (DDPG).

    Instead of popular Ornstein-Uhlenbeck (OU) process for noise this agent uses Gaussian noise.
    """

    name = "DDPG"

    def __init__(
        self, state_size: int, action_size: int, hidden_layers: Sequence[int]=(128, 128),
        actor_lr: float=2e-3, actor_lr_decay: float=0, critic_lr: float=2e-3, critic_lr_decay: float=0,
        noise_scale: float=0.2, noise_sigma: float=0.1, clip: Tuple[int, int]=(-1, 1), **kwargs
    ):
        self.device = device = kwargs.get("device", DEVICE)

        # Reason sequence initiation.
        self.hidden_layers = kwargs.get('hidden_layers', hidden_layers)
        self.actor = ActorBody(state_size, action_size, hidden_layers=hidden_layers).to(self.device)
        self.critic = CriticBody(state_size, action_size, hidden_layers=hidden_layers).to(self.device)
        self.target_actor = ActorBody(state_size, action_size, hidden_layers=hidden_layers).to(self.device)
        self.target_critic = CriticBody(state_size, action_size, hidden_layers=hidden_layers).to(self.device)

        # Noise sequence initiation
        self.noise = GaussianNoise(shape=(action_size,), mu=1e-8, sigma=noise_sigma, scale=noise_scale, device=device)

        # Target sequence initiation
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        # Optimization sequence initiation.
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr, weight_decay=actor_lr_decay)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr, weight_decay=critic_lr_decay)
        self.max_grad_norm_actor: float = float(kwargs.get("max_grad_norm_actor", 10.0))
        self.max_grad_norm_critic: float = float(kwargs.get("max_grad_norm_critic", 10.0))
        self.action_min = clip[0]
        self.action_max = clip[1]
        self.action_scale = kwargs.get('action_scale', 1)

        self.gamma: float = float(kwargs.get('gamma', 0.99))
        self.tau: float = float(kwargs.get('tau', 0.02))
        self.batch_size: int = int(kwargs.get('batch_size', 64))
        self.buffer_size: int = int(kwargs.get('buffer_size', int(1e6)))
        self.buffer = ReplayBuffer(self.batch_size, self.buffer_size)

        self.warm_up: int = int(kwargs.get('warm_up', 0))
        self.update_freq: int = int(kwargs.get('update_freq', 1))
        self.number_updates: int = int(kwargs.get('number_updates', 1))

        # Breath, my child.
        self.reset_agent()
        self.iteration = 0

    def reset_agent(self) -> None:
        self.actor.reset_parameters()
        self.critic.reset_parameters()
        self.target_actor.reset_parameters()
        self.target_critic.reset_parameters()

    def act(self, obs, noise: float=0.0):
        with torch.no_grad():
            obs = to_tensor(obs).float().to(self.device)
            action = self.actor(obs)
            action += noise*self.noise.sample()
            return self.action_scale*torch.clamp(action, self.action_min, self.action_max).cpu().numpy().astype(_float32)

    def target_act(self, obs, noise: float=0.0):
        with torch.no_grad():
            obs = to_tensor(obs).float().to(self.device)
            action = self.target_actor(obs) + noise*self.noise.sample()
            return torch.clamp(action, self.action_min, self.action_max).cpu().numpy().astype(_float32)

    def step(self, state, action, reward, next_state, done):
        self.iteration += 1
        self.buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

        if self.iteration < self.warm_up:
            return

        if len(self.buffer) > self.batch_size and (self.iteration % self.update_freq) == 0:
            for _ in range(self.number_updates):
                self.learn(self.buffer.sample())

    def learn(self, experiences):
        """Update critics and actors"""
        rewards = to_tensor(experiences['reward']).float().to(self.device).unsqueeze(1)
        dones = to_tensor(experiences['done']).type(torch.int).to(self.device).unsqueeze(1)
        states = to_tensor(experiences['state']).float().to(self.device)
        actions = to_tensor(experiences['action']).to(self.device)
        next_states = to_tensor(experiences['next_state']).float().to(self.device)

        # critic loss
        next_actions = self.target_actor(next_states)
        Q_target_next = self.target_critic(next_states, next_actions)
        Q_target = rewards + (self.gamma * Q_target_next * (1 - dones))
        Q_expected = self.critic(states, actions)
        critic_loss = mse_loss(Q_expected, Q_target)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm_critic)
        self.critic_optimizer.step()
        self.critic_loss = critic_loss.item()

        # Compute actor loss
        pred_actions = self.actor(states)
        actor_loss = -self.critic(states, pred_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm_critic)
        self.actor_optimizer.step()
        self.actor_loss = actor_loss.item()

        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)

    def describe_agent(self) -> Tuple[Any, Any, Any, Any]:
        """
        Returns network's weights in order:
        Actor, TargetActor, Critic, TargetCritic
        """
        return (self.actor.state_dict(), self.target_actor.state_dict(), self.critic.state_dict(), self.target_critic())

    def log_writer(self, writer, episode):
        writer.add_scalar("loss/actor", self.actor_loss, episode)
        writer.add_scalar("loss/critic", self.critic_loss, episode)

    def save_state(self, path: str):
        agent_state = dict(
            actor=self.actor.state_dict(), target_actor=self.target_actor.state_dict(),
            critic=self.critic.state_dict(), target_critic=self.target_critic.state_dict(),
        )
        torch.save(agent_state, path)

    def load_state(self, path: str):
        agent_state = torch.load(path)
        self.actor.load_state_dict(agent_state['actor'])
        self.critic.load_state_dict(agent_state['critic'])
        self.target_actor.load_state_dict(agent_state['target_actor'])
        self.target_critic.load_state_dict(agent_state['target_critic'])
