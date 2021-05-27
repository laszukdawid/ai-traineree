import itertools
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_traineree import DEVICE
from ai_traineree.agents import AgentBase
from ai_traineree.agents.agent_utils import hard_update, soft_update
from ai_traineree.buffers import NStepBuffer, PERBuffer
from ai_traineree.loggers import DataLogger
from ai_traineree.networks.bodies import ActorBody, CriticBody
from ai_traineree.networks.heads import CategoricalNet
from ai_traineree.policies import MultivariateGaussianPolicy, MultivariateGaussianPolicySimple
from ai_traineree.utils import to_numbers_seq, to_tensor
from torch.optim import Adam


class D4PGAgent(AgentBase):
    """
    Distributed Distributional DDPG (D4PG) [1].

    Extends the DDPG agent with:
    1. Distributional critic update.
    2. The use of distributed parallel actors.
    3. N-step returns.
    4. Prioritization of the experience replay (PER).

    [1] "Distributed Distributional Deterministic Policy Gradients"
        (2018, ICLR) by G. Barth-Maron & M. Hoffman et al.

    """

    name = "D4PG"

    def __init__(self, state_size: int, action_size: int, hidden_layers: Sequence[int]=(128, 128), **kwargs):
        """
        Parameters:
            state_size (int): Number of input dimensions.
            action_size (int): Number of output dimensions
            hidden_layers (tuple of ints): Tuple defining hidden dimensions in fully connected nets. Default: (128, 128).

        Keyword parameters:
            gamma (float): Discount value. Default: 0.99.
            tau (float): Soft-copy factor. Default: 0.02.
            actor_lr (float): Learning rate for the actor (policy). Default: 0.0003.
            critic_lr (float): Learning rate for the critic (value function). Default: 0.0003.
            actor_hidden_layers (tuple of ints): Shape of network for actor. Default: `hideen_layers`.
            critic_hidden_layers (tuple of ints): Shape of network for critic. Default: `hideen_layers`.
            max_grad_norm_actor (float) Maximum norm value for actor gradient. Default: 100.
            max_grad_norm_critic (float): Maximum norm value for critic gradient. Default: 100.
            num_atoms (int): Number of discrete values for the value distribution. Default: 51.
            v_min (float): Value distribution minimum (left most) value. Default: -10.
            v_max (float): Value distribution maximum (right most) value. Default: 10.
            n_steps (int): Number of steps (N-steps) for the TD. Defualt: 3.
            batch_size (int): Number of samples used in learning. Default: 64.
            buffer_size (int): Maximum number of samples to store. Default: 1e6.
            warm_up (int): Number of samples to observe before starting any learning step. Default: 0.
            update_freq (int): Number of steps between each learning step. Default 1.
            number_updates (int): How many times to use learning step in the learning phase. Default: 1.
            action_min (float): Minimum returned action value. Default: -1.
            action_max (float): Maximum returned action value. Default: 1.
            action_scale (float): Multipler value for action. Default: 1.
            num_workers (int): Number of workers that will assume this agent. Default: 1.

        """
        super().__init__(**kwargs)
        self.device = self._register_param(kwargs, "device", DEVICE)
        self.state_size = state_size
        self.action_size = action_size

        self.num_atoms = int(self._register_param(kwargs, 'num_atoms', 51))
        v_min = float(self._register_param(kwargs, 'v_min', -10))
        v_max = float(self._register_param(kwargs, 'v_max', 10))

        # Reason sequence initiation.
        self.action_min = float(self._register_param(kwargs, 'action_min', -1))
        self.action_max = float(self._register_param(kwargs, 'action_max', 1))
        self.action_scale = float(self._register_param(kwargs, 'action_scale', 1))

        self.gamma = float(self._register_param(kwargs, 'gamma', 0.99))
        self.tau = float(self._register_param(kwargs, 'tau', 0.02))
        self.batch_size = int(self._register_param(kwargs, 'batch_size', 64))
        self.buffer_size = int(self._register_param(kwargs, 'buffer_size', int(1e6)))
        self.buffer = PERBuffer(self.batch_size, self.buffer_size)

        self.n_steps = int(self._register_param(kwargs, "n_steps", 3))
        self.n_buffer = NStepBuffer(n_steps=self.n_steps, gamma=self.gamma)

        self.warm_up = int(self._register_param(kwargs, 'warm_up', 0))
        self.update_freq = int(self._register_param(kwargs, 'update_freq', 1))

        self.actor_hidden_layers = to_numbers_seq(self._register_param(kwargs, 'actor_hidden_layers', hidden_layers))
        self.critic_hidden_layers = to_numbers_seq(self._register_param(kwargs, 'critic_hidden_layers', hidden_layers))

        if kwargs.get("simple_policy", False):
            std_init = float(self._register_param(kwargs, "std_init", 1.0))
            std_max = float(self._register_param(kwargs, "std_max", 2.0))
            std_min = float(self._register_param(kwargs, "std_min", 0.05))
            self.policy = MultivariateGaussianPolicySimple(self.action_size, std_init=std_init, std_min=std_min, std_max=std_max, device=self.device)
        else:
            self.policy = MultivariateGaussianPolicy(self.action_size, device=self.device)

        # This looks messy but it's not that bad. Actor, critic_net and Critic(critic_net). Then the same for `target_`.
        self.actor = ActorBody(
            state_size, self.policy.param_dim*action_size, hidden_layers=self.actor_hidden_layers,
            gate_out=torch.tanh, device=self.device
        )
        critic_net = CriticBody(
            state_size, action_size, out_features=self.num_atoms, hidden_layers=self.critic_hidden_layers, device=self.device
        )
        self.critic = CategoricalNet(
            num_atoms=self.num_atoms, v_min=v_min, v_max=v_max, net=critic_net, device=self.device
        )

        self.target_actor = ActorBody(
            state_size, self.policy.param_dim*action_size, hidden_layers=self.actor_hidden_layers,
            gate_out=torch.tanh, device=self.device
        )
        target_critic_net = CriticBody(
            state_size, action_size, out_features=self.num_atoms, hidden_layers=self.critic_hidden_layers, device=self.device
        )
        self.target_critic = CategoricalNet(
            num_atoms=self.num_atoms, v_min=v_min, v_max=v_max, net=target_critic_net, device=self.device
        )

        # Target sequence initiation
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        # Optimization sequence initiation.
        self.actor_lr = float(self._register_param(kwargs, 'actor_lr', 3e-4))
        self.critic_lr = float(self._register_param(kwargs, 'critic_lr', 3e-4))
        self.value_loss_func = nn.BCELoss(reduction='none')

        self.actor_params = list(self.actor.parameters()) + list(self.policy.parameters())
        self.actor_optimizer = Adam(self.actor_params, lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        self.max_grad_norm_actor = float(self._register_param(kwargs, "max_grad_norm_actor", 100))
        self.max_grad_norm_critic = float(self._register_param(kwargs, "max_grad_norm_critic", 100))

        self.num_workers = int(self._register_param(kwargs, "num_workers", 1))

        # Breath, my child.
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

    @torch.no_grad()
    def act(self, state, epsilon: float=0.) -> List[float]:
        """
        Returns actions for given state as per current policy.

        Parameters:
            state: Current available state from the environment.
            epislon: Epsilon value in the epislon-greedy policy.

        """
        actions = []
        state = to_tensor(state).view(self.num_workers, self.state_size).float().to(self.device)
        for worker in range(self.num_workers):
            if self._rng.random() < epsilon:
                action = self.action_scale*(torch.rand(self.action_size) - 0.5)
            else:
                action_seed = self.actor.act(state[worker].view(1, -1))
                action_dist = self.policy(action_seed)
                action = action_dist.sample()
                action *= self.action_scale
                action = torch.clamp(action.squeeze(), self.action_min, self.action_max).cpu()
            actions.append(action.tolist())

        assert len(actions) == self.num_workers
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        self.iteration += 1

        # Delay adding to buffer to account for n_steps (particularly the reward)
        self.n_buffer.add(
            state=torch.tensor(states).reshape(self.num_workers, self.state_size).float(),
            next_state=torch.tensor(next_states).reshape(self.num_workers, self.state_size).float(),
            action=torch.tensor(actions).reshape(self.num_workers, self.action_size).float(),
            reward=torch.tensor(rewards).reshape(self.num_workers, 1),
            done=torch.tensor(dones).reshape(self.num_workers, 1),
        )
        if not self.n_buffer.available:
            return

        samples = self.n_buffer.get().get_dict()
        for worker_idx in range(self.num_workers):
            self.buffer.add(
                state=samples['state'][worker_idx],
                next_state=samples['next_state'][worker_idx],
                action=samples['action'][worker_idx],
                done=samples['done'][worker_idx],
                reward=samples['reward'][worker_idx],
            )

        if self.iteration < self.warm_up:
            return

        if len(self.buffer) > self.batch_size and (self.iteration % self.update_freq) == 0:
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
        self._metric_batch_error = loss.detach().sum(dim=-1)
        samples_error = loss.sum(dim=-1).pow(2)
        loss_critic = samples_error.mean()

        if hasattr(self.buffer, 'priority_update') and indices is not None:
            assert (~torch.isnan(samples_error)).any()
            self.buffer.priority_update(indices, samples_error.detach().cpu().numpy())

        return loss_critic

    def compute_policy_loss(self, states):
        # Compute actor loss
        pred_action_seeds = self.actor(states)
        pred_actions = self.policy.act(pred_action_seeds)
        pred_actions = self.policy(pred_action_seeds).rsample()
        # Negative because the optimizer minimizes, but we want to maximize the value
        value_dist = self.critic(states, pred_actions)
        self._batch_value_dist_metric = value_dist.detach()
        # Estimate on Z support
        return -torch.mean(value_dist*self.critic.z_atoms)

    def learn(self, experiences):
        """Update critics and actors"""
        # No need for size assertion since .view() has explicit sizes
        rewards = to_tensor(experiences['reward']).view(self.batch_size, 1).float().to(self.device)
        dones = to_tensor(experiences['done']).view(self.batch_size, 1).type(torch.int).to(self.device)
        states = to_tensor(experiences['state']).view(self.batch_size, self.state_size).float().to(self.device)
        actions = to_tensor(experiences['action']).view(self.batch_size, self.action_size).to(self.device)
        next_states = to_tensor(experiences['next_state']).view(self.batch_size, self.state_size).float().to(self.device)

        indices = None
        if hasattr(self.buffer, 'priority_update'):  # When using PER buffer
            indices = experiences['index']

        # Value (critic) optimization
        loss_critic = self.compute_value_loss(states, actions, next_states, rewards, dones, indices)
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

    def log_metrics(self, data_logger: DataLogger, step, full_log=False):
        data_logger.log_value("loss/actor", self._loss_actor, step)
        data_logger.log_value("loss/critic", self._loss_critic, step)
        policy_params = {str(i): v for i, v in enumerate(itertools.chain.from_iterable(self.policy.parameters()))}
        data_logger.log_values_dict("policy/param", policy_params, step)

        data_logger.create_histogram('metric/batch_errors', self._metric_batch_error.sum(-1), step)
        data_logger.create_histogram('metric/batch_value_dist', self._batch_value_dist_metric, step)

        # This method, `log_metrics`, isn't executed on every iteration but just in case we delay plotting weights.
        # It simply might be quite costly. Thread wisely.
        if full_log:
            for idx, layer in enumerate(self.actor.layers):
                if hasattr(layer, "weight"):
                    data_logger.create_histogram(f"actor/layer_weights_{idx}", layer.weight, step)
                if hasattr(layer, "bias") and layer.bias is not None:
                    data_logger.create_histogram(f"actor/layer_bias_{idx}", layer.bias, step)

            for idx, layer in enumerate(self.critic.net.layers):
                if hasattr(layer, "weight"):
                    data_logger.create_histogram(f"critic/layer_{idx}", layer.weight, step)
                if hasattr(layer, "bias") and layer.bias is not None:
                    data_logger.create_histogram(f"critic/layer_bias_{idx}", layer.bias, step)

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
