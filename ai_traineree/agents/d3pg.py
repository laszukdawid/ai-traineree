import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from ai_traineree import DEVICE
from ai_traineree.agents.utils import hard_update, soft_update
from ai_traineree.buffers import NStepBuffer, PERBuffer
from ai_traineree.networks.bodies import ActorBody, CriticBody
from ai_traineree.networks.heads import CategoricalNet
from ai_traineree.policies import MultivariateGaussianPolicySimple, MultivariateGaussianPolicy
from ai_traineree.types import AgentType
from ai_traineree.utils import to_tensor
from torch.optim import Adam
from typing import Any, Dict, Sequence, Tuple


class D3PGAgent(AgentType):
    """Distributional DDPG (D3PG) [1].

    It's closely related to, and sits in-between, D4PG and DDPG. Compared to D4PG it lacks
    the multi actors support. It extends the DDPG agent with:
    1. Distributional critic update.
    2. N-step returns.
    3. Prioritization of the experience replay (PER).

    [1] "Distributed Distributional Deterministic Policy Gradients"
        (2018, ICLR) by G. Barth-Maron & M. Hoffman et al. 

    """

    name = "D3PG"

    def __init__(
        self, state_size: int,
        action_size: int,
        hidden_layers: Sequence[int]=(128, 128),
        actor_lr: float=2e-4,
        critic_lr: float=2e-4,
        clip: Tuple[int, int]=(-1, 1),
        **kwargs
    ):
        self.device = self._register_param(kwargs, "device", DEVICE)
        self.state_size = state_size
        self.action_size = action_size

        self.num_atoms = int(self._register_param(kwargs, 'num_atoms', 51))
        v_min = float(self._register_param(kwargs, 'v_min', -10))
        v_max = float(self._register_param(kwargs, 'v_max', 10))

        # Reason sequence initiation.
        self.action_min = clip[0]
        self.action_max = clip[1]
        self.action_scale = self._register_param(kwargs, 'action_scale', 1)

        self.gamma: float = float(self._register_param(kwargs, 'gamma', 0.99))
        self.tau: float = float(self._register_param(kwargs, 'tau', 0.02))
        self.batch_size: int = int(self._register_param(kwargs, 'batch_size', 64))
        self.buffer_size: int = int(self._register_param(kwargs, 'buffer_size', int(1e6)))
        self.buffer = PERBuffer(self.batch_size, self.buffer_size)
        # self.buffer = ReplayBuffer(self.batch_size, self.buffer_size)

        self.n_steps = self._register_param(kwargs, "n_steps", 3)
        self.n_buffer = NStepBuffer(n_steps=self.n_steps, gamma=self.gamma)

        self.warm_up: int = int(self._register_param(kwargs, 'warm_up', 0))
        self.update_freq: int = int(self._register_param(kwargs, 'update_freq', 1))
        self.number_updates: int = int(self._register_param(kwargs, 'number_updates', 1))

        # self.hidden_layers = self._register_param(kwargs, 'hidden_layers', hidden_layers)
        self.hidden_layers = kwargs.get('hidden_layers', hidden_layers)

        if kwargs.get("simple_policy", False):
            std_init = kwargs.get("std_init", 1.0)
            std_max = kwargs.get("std_max", 1.5)
            std_min = kwargs.get("std_min", 0.25)
            self.policy = MultivariateGaussianPolicySimple(self.action_size, std_init=std_init, std_min=std_min, std_max=std_max, device=self.device)
        else:
            self.policy = MultivariateGaussianPolicy(self.action_size, device=self.device)

        self.actor = ActorBody(state_size, self.policy.param_dim*action_size, hidden_layers=hidden_layers, gate_out=torch.tanh, device=self.device)
        critic = CriticBody(state_size, action_size, out_features=self.num_atoms, hidden_layers=hidden_layers, device=self.device)
        self.critic = CategoricalNet(
            state_size + action_size, 1, num_atoms=self.num_atoms,
            v_min=v_min, v_max=v_max, hidden_layers=hidden_layers, net=critic, device=self.device
        )
        self.target_actor = ActorBody(state_size, self.policy.param_dim*action_size, hidden_layers=hidden_layers, gate_out=torch.tanh, device=self.device)
        target_critic = CriticBody(state_size, action_size, out_features=self.num_atoms, hidden_layers=hidden_layers, device=self.device)
        self.target_critic = CategoricalNet(
            state_size + action_size, 1, num_atoms=self.num_atoms, v_min=v_min, v_max=v_max,
            net=target_critic, hidden_layers=hidden_layers, device=self.device
        )
        self.value_loss_func = nn.BCELoss(reduction='none')

        # Target sequence initiation
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        # Optimization sequence initiation.
        self.actor_lr = self._register_param(kwargs, 'actor_lr', actor_lr)
        self.critic_lr = self._register_param(kwargs, 'critic_lr', critic_lr)

        # self.actor_params = list(self.actor.parameters()) #+ list(self.policy.parameters())
        self.actor_params = list(self.actor.parameters()) + list(self.policy.parameters())
        self.actor_optimizer = Adam(self.actor_params, lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        self.max_grad_norm_actor: float = float(self._register_param(kwargs, "max_grad_norm_actor", 50.0))
        self.max_grad_norm_critic: float = float(self._register_param(kwargs, "max_grad_norm_critic", 50.0))

        # Breath, my child.
        self.reset_agent()
        self.iteration = 0
        self._loss_actor = float('nan')
        self._loss_critic = float('nan')
        self._display_dist = torch.empty(self.critic.z_atoms.shape)

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

    @torch.no_grad()
    def act(self, obs, eps: float=0.0):
        """
        In the paper they used epsilon = 0.3.
        """
        obs = to_tensor(obs).float().to(self.device)
        if random.random() < eps:
            action = self.action_scale*(torch.rand(self.action_size) - 0.5)

        else:
            action_seed = self.actor.act(obs).view(1, -1)
            action_dist = self.policy(action_seed)
            action = action_dist.sample()
            action *= self.action_scale
            action = action.squeeze()

        # Purely for logging
        self._display_dist = self.target_critic.act(obs, action.to(self.device)).squeeze().cpu()
        self._display_dist = F.softmax(self._display_dist, dim=0)

        return torch.clamp(action, self.action_min, self.action_max).cpu().numpy()

    @torch.no_grad()
    def target_act(self, obs, eps: float=0.0):
        if random.random() < eps:
            random_action = torch.rand(self.action_size)*(self.action_max - self.action_min) + self.action_min
            return random_action.numpy()

        obs = to_tensor(obs).float().to(self.device)
        action = self.target_actor(obs)
        return torch.clamp(action, self.action_min, self.action_max).cpu().numpy()

    def step(self, state, action, reward, next_state, done):
        self.iteration += 1

        # Delay adding to buffer to account for n_steps (particularly the reward)
        self.n_buffer.add(state=state, action=action, reward=[reward], done=[done], next_state=next_state)
        if not self.n_buffer.available:
            return

        self.buffer.add(**self.n_buffer.get().get_dict())

        if self.iteration < self.warm_up:
            return

        if len(self.buffer) > self.batch_size and (self.iteration % self.update_freq) == 0:
            for _ in range(self.number_updates):
                self.learn(self.buffer.sample())

    def compute_value_loss(self, states, actions, next_states, rewards, dones, indices=None):
        # Q_w estimate
        value_dist_estimate = self.critic(states, actions)
        assert value_dist_estimate.shape == (self.batch_size, 1, self.num_atoms)
        value_dist = F.softmax(value_dist_estimate.squeeze(), dim=1)
        assert value_dist.shape == (self.batch_size, self.num_atoms)

        # Q_w' estimate via Bellman's dist operator
        next_action_seeds = self.target_actor.act(next_states)
        next_actions = self.policy(next_action_seeds).sample()
        assert next_actions.shape == (self.batch_size, self.action_size)

        target_value_dist_estimate = self.target_critic.act(states, next_actions)
        assert target_value_dist_estimate.shape == (self.batch_size, 1, self.num_atoms)
        target_value_dist_estimate = target_value_dist_estimate.squeeze()
        assert target_value_dist_estimate.shape == (self.batch_size, self.num_atoms)

        discount = self.gamma ** self.n_steps
        target_value_projected = self.target_critic.dist_projection(rewards, 1 - dones, discount, target_value_dist_estimate)
        assert target_value_projected.shape == (self.batch_size, self.num_atoms)

        target_value_dist = F.softmax(target_value_dist_estimate, dim=-1).detach()
        assert target_value_dist.shape == (self.batch_size, self.num_atoms)

        # Comparing Q_w with Q_w'
        loss = self.value_loss_func(value_dist, target_value_projected)
        samples_error = loss.sum(dim=-1).abs()  # TODO: Is this good metric?
        loss_critic = samples_error.mean()

        if hasattr(self.buffer, 'priority_update') and indices is not None:
            assert (~torch.isnan(samples_error)).any()
            self.buffer.priority_update(indices, samples_error.detach().cpu().numpy())

        return loss_critic

    def compute_policy_loss(self, states):
        # Compute actor loss
        pred_action_seeds = self.actor(states)
        pred_actions = self.policy(pred_action_seeds).rsample()
        # Negative because the optimizer minimizes, but we want to maximize the value
        value_dist = self.critic(states, pred_actions)
        # Estimate on Z support
        return -torch.mean(value_dist*self.critic.z_atoms)

    def learn(self, experiences):
        """Update critics and actors"""
        rewards = to_tensor(experiences['reward']).float().to(self.device)
        dones = to_tensor(experiences['done']).type(torch.int).to(self.device)
        states = to_tensor(experiences['state']).float().to(self.device)
        actions = to_tensor(experiences['action']).to(self.device)
        next_states = to_tensor(experiences['next_state']).float().to(self.device)
        assert rewards.shape == dones.shape == (self.batch_size, 1)
        assert states.shape == next_states.shape == (self.batch_size, self.state_size)
        assert actions.shape == (self.batch_size, self.action_size)

        indices = None
        if hasattr(self.buffer, 'priority_update'):  # When using PER buffer
            indices = experiences['index']
        loss_critic = self.compute_value_loss(states, actions, next_states, rewards, dones, indices)

        # Value (critic) optimization
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm_critic)
        self.critic_optimizer.step()
        self._loss_critic = float(loss_critic.item())

        # Policy (actor) optimization
        loss_actor = self.compute_policy_loss(states)
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm_actor)
        self.actor_optimizer.step()
        self._loss_actor = float(loss_actor.item())

        # Networks gradual sync
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)

    def describe_agent(self) -> Tuple[Any, Any, Any, Any]:
        """
        Returns network's weights in order:
        Actor, TargetActor, Critic, TargetCritic
        """
        return (self.actor.state_dict(), self.target_actor.state_dict(), self.critic.state_dict(), self.target_critic())

    def log_writer(self, writer, step):
        writer.add_scalar("loss/actor", self._loss_actor, step)
        writer.add_scalar("loss/critic", self._loss_critic, step)
        policy_params = {str(i): v for i, v in enumerate(itertools.chain.from_iterable(self.policy.parameters()))}
        writer.add_scalars("policy/param", policy_params, step)

        dist = self._display_dist
        z_atoms = self.critic.z_atoms
        z_delta = self.critic.z_delta
        writer.add_histogram_raw(
            'dist/dist_value', min=z_atoms[0], max=z_atoms[-1], num=self.num_atoms,
            sum=dist.sum(), sum_squares=dist.pow(2).sum(), bucket_limits=z_atoms+z_delta,
            bucket_counts=dist, global_step=step
        )

    def save_state(self, path: str):
        agent_state = dict(
            actor=self.actor.state_dict(), target_actor=self.target_actor.state_dict(),
            critic=self.critic.state_dict(), target_critic=self.target_critic.state_dict(),
            config=self._config,
        )
        torch.save(agent_state, path)

    def load_state(self, path: str):
        agent_state = torch.load(path)
        self._config = agent_state.get('config', {})
        self.__dict__.update(**self._config)

        self.actor.load_state_dict(agent_state['actor'])
        self.critic.load_state_dict(agent_state['critic'])
        self.target_actor.load_state_dict(agent_state['target_actor'])
        self.target_critic.load_state_dict(agent_state['target_critic'])
