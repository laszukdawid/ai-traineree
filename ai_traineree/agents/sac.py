import copy
import itertools
from functools import cached_property
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, optim

from ai_traineree import DEVICE
from ai_traineree.agents import AgentBase
from ai_traineree.agents.agent_utils import hard_update, soft_update
from ai_traineree.buffers import PERBuffer
from ai_traineree.buffers.buffer_factory import BufferFactory
from ai_traineree.loggers import DataLogger
from ai_traineree.networks.bodies import ActorBody, CriticBody
from ai_traineree.networks.heads import DoubleCritic
from ai_traineree.policies import GaussianPolicy, MultivariateGaussianPolicySimple
from ai_traineree.types.dataspace import DataSpace
from ai_traineree.types.experience import Experience
from ai_traineree.types.state import AgentState, BufferState, NetworkState
from ai_traineree.utils import to_numbers_seq, to_tensor


class SACAgent(AgentBase):
    """
    Soft Actor-Critic.

    Uses stochastic policy and dual value network (two critics).

    Based on
    "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    by Haarnoja et al. (2018) (http://arxiv.org/abs/1801.01290).

    """

    model = "SAC"

    def __init__(self, obs_space: DataSpace, action_space: DataSpace, **kwargs):
        """
        Parameters:
            obs_space (DataSpace): Dataspace describing the input.
            action_space (DataSpace): Dataspace describing the output.

        Keyword arguments:
            hidden_layers (tuple of ints): Shape of the hidden layers in fully connected network. Default: (128, 128).
            gamma (float):  Discount value. Default: 0.99.
            tau (float): Soft copy fraction. Default: 0.02.
            batch_size (int): Number of samples in a batch. Default: 64.
            buffer_size (int): Size of the prioritized experience replay buffer. Default: 1e6.
            warm_up: (default: 0) Number of samples that needs to be observed before starting to learn. Default: 0.
            update_freq (int): Number of samples between policy updates. Default: 1.
            number_updates (int):  Number of times of batch sampling/training per `update_freq`. Default: 1.
            alpha (float): Weight of log probs in value function. Default: 0.2.
            alpha_lr (Optional float): If not None, it adds alpha as a training parameters with `alpha_lr` as its
                learning rate. Default: None.
            action_scale (float):  Scale for returned action values. Default: 1.
            max_grad_norm_alpha (float): Gradient clipping for the alpha. Default: 1.
            max_grad_norm_actor (float): Gradient clipping for the actor. Default: 10.
            max_grad_norm_critic (float):  Gradient clipping for the critic. Default: 10.
            device: Defaults to CUDA if available. Default: CUDA if available.

        """
        super().__init__(**kwargs)
        self.device = kwargs.get("device", DEVICE)
        self.obs_space = obs_space
        self.action_space = action_space
        self._config["obs_space"] = self.obs_space
        self._config["action_space"] = self.action_space
        action_size = self.action_space.shape[0]  # Because of 1D

        self.gamma: float = float(self._register_param(kwargs, "gamma", 0.99))
        self.tau: float = float(self._register_param(kwargs, "tau", 0.02))
        self.batch_size: int = int(self._register_param(kwargs, "batch_size", 64))
        self.buffer_size: int = int(self._register_param(kwargs, "buffer_size", int(1e6)))
        self.buffer = PERBuffer(self.batch_size, self.buffer_size)

        self.action_scale = self._register_param(kwargs, "action_scale", 1)

        self.warm_up = int(self._register_param(kwargs, "warm_up", 0))
        self.update_freq = int(self._register_param(kwargs, "update_freq", 1))
        self.number_updates = int(self._register_param(kwargs, "number_updates", 1))
        self.critic_number_updates = int(self._register_param(kwargs, "critic_number_updates", 1))
        self.actor_number_updates = int(self._register_param(kwargs, "actor_number_updates", 1))

        # Reason sequence initiation.
        hidden_layers = to_numbers_seq(self._register_param(kwargs, "hidden_layers", (128, 128)))
        actor_hidden_layers = to_numbers_seq(self._register_param(kwargs, "actor_hidden_layers", hidden_layers))
        critic_hidden_layers = to_numbers_seq(self._register_param(kwargs, "critic_hidden_layers", hidden_layers))

        self.simple_policy = bool(self._register_param(kwargs, "simple_policy", True))
        if self.simple_policy:
            self.policy = MultivariateGaussianPolicySimple(action_size, **kwargs)
            self.actor = ActorBody(
                obs_space.shape,
                (self.policy.param_dim * action_size,),
                hidden_layers=actor_hidden_layers,
                gate=nn.ReLU(),
                gate_out=torch.tanh,
                device=self.device,
            )
        else:
            self.policy = GaussianPolicy(
                (actor_hidden_layers[-1],),
                self.action_space.shape,
                out_scale=self.action_scale,
                gate=nn.ReLU(),
                device=self.device,
            )
            self.actor = ActorBody(
                obs_space.shape,
                (actor_hidden_layers[-1],),
                hidden_layers=actor_hidden_layers[:-1],
                gate=nn.ReLU(),
                gate_out=torch.tanh,
                device=self.device,
            )

        self.double_critic = DoubleCritic(
            obs_space.shape,
            action_size,
            CriticBody,
            hidden_layers=critic_hidden_layers,
            gate=nn.ReLU(),
            device=self.device,
        )
        self.target_double_critic = DoubleCritic(
            obs_space.shape,
            action_size,
            CriticBody,
            hidden_layers=critic_hidden_layers,
            gate=nn.ReLU(),
            device=self.device,
        )

        # Target sequence initiation
        hard_update(self.target_double_critic, self.double_critic)

        # Optimization sequence initiation.
        self.target_entropy = -action_size
        alpha_lr = self._register_param(kwargs, "alpha_lr")
        self.alpha_lr = float(alpha_lr) if alpha_lr else None
        alpha_init = float(self._register_param(kwargs, "alpha", 0.2))
        self.log_alpha = torch.tensor(np.log(alpha_init), device=self.device, requires_grad=True)

        actor_lr = float(self._register_param(kwargs, "actor_lr", 3e-4))
        critic_lr = float(self._register_param(kwargs, "critic_lr", 3e-4))

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

        self._loss_actor = float("nan")
        self._loss_critic = float("nan")
        self._metrics: Dict[str, Union[float, Dict[str, float]]] = {}

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def loss(self) -> Dict[str, float]:
        return {"actor": self._loss_actor, "critic": self._loss_critic}

    @loss.setter
    def loss(self, value):
        if isinstance(value, dict):
            self._loss_actor = value["actor"]
            self._loss_critic = value["critic"]
        else:
            self._loss_actor = value
            self._loss_critic = value

    def __eq__(self, o: object) -> bool:
        return (
            super().__eq__(o)
            and isinstance(o, type(self))
            and self._config == o._config
            and self.buffer == o.buffer
            and self.get_network_state() == o.get_network_state()  # TODO @dawid: Currently net isn't compared properly
        )

    def reset_agent(self) -> None:
        self.actor.reset_parameters()
        self.policy.reset_parameters()
        self.double_critic.reset_parameters()
        hard_update(self.target_double_critic, self.double_critic)

    @cached_property
    def action_min(self):
        return to_tensor(self.action_space.low)

    @cached_property
    def action_max(self):
        return to_tensor(self.action_space.high)

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
    def act(self, experience: Experience, epsilon: float = 0.0, deterministic: bool = False) -> Experience:
        """Acting on the observations. Returns action.

        Parameters:
            obs (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
            deterministic (optional bool): Whether to use deterministic policy. Only has effect in `train` mode.
                In `test` mode all actions are deterministic.

        Returns:
            action: (list float) Action values.

        """
        if self.train and self.iteration < self.warm_up or self._rng.random() < epsilon:
            rnd = torch.rand(self.action_space.shape)
            rnd_action = (self.action_max + self.action_min) * rnd + self.action_min
            action = rnd_action.cpu().tolist()
            return experience.update(action=action)

        _deterministic = (not self.train) or deterministic
        t_obs = to_tensor(experience.obs).view((1,) + self.obs_space.shape).float().to(self.device)

        proto_action = self.actor(t_obs)
        action = self.policy(proto_action, _deterministic).flatten()
        if self.train and not deterministic:
            last_samples = self.policy._last_samples
            added_noise = action - self.policy(proto_action, deterministic=True).flatten()
            noise_params = [*last_samples.get("mu").flatten().tolist(), *last_samples.get("std").flatten().tolist()]
            experience = experience.update(noise=added_noise, noise_params=noise_params)
        action = torch.clamp(action, self.action_min, self.action_max)
        action = action.tolist()
        return experience.update(action=action)

    def step(self, experience: Experience) -> None:
        if not self.train:
            return

        self.iteration += 1
        self.buffer.add(
            obs=experience.obs,
            action=experience.action,
            reward=experience.reward,
            next_obs=experience.next_obs,
            done=experience.done,
        )

        if self.iteration < self.warm_up:
            return

        if len(self.buffer) > self.batch_size and (self.iteration % self.update_freq) == 0:
            for _ in range(self.number_updates):
                self.learn(self.buffer.sample())

    def compute_value_loss(self, states, actions, rewards, next_states, dones) -> Tuple[Tensor, Tensor]:
        Q1_expected, Q2_expected = self.double_critic(states, actions)

        with torch.no_grad():
            # next_actions, log_prob = self.actor(next_states)
            proto_next_action = self.actor(next_states)
            next_actions = self.policy(proto_next_action)
            log_prob = self.policy.log_prob(next_actions).view(-1, 1)
            assert next_actions.shape == (self.batch_size,) + self.action_space.shape
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
        mse_loss_1: Tensor = error_1.mean()
        self._metrics["value/critic1"] = {"mean": float(Q1_expected.mean()), "std": float(Q1_expected.std())}
        self._metrics["value/critic1_lse"] = float(mse_loss_1.item())

        Q2_diff = Q2_expected - Q_target
        error_2 = Q2_diff.pow(2)
        mse_loss_2: Tensor = error_2.mean()
        self._metrics["value/critic2"] = {"mean": float(Q2_expected.mean()), "std": float(Q2_expected.std())}
        self._metrics["value/critic2_lse"] = float(mse_loss_2.item())

        with torch.no_grad():
            Q_diff = Q1_expected - Q2_expected
            self._metrics["value/Q_diff"] = {"mean": float(Q_diff.mean()), "std": float(Q_diff.std())}
            error: Tensor = torch.max(error_1, error_2)

        loss = mse_loss_1 + mse_loss_2
        return loss, error

    def compute_policy_loss(self, states):
        self.double_critic.requires_grad_ = False
        proto_actions = self.actor(states)
        pred_actions = self.policy(proto_actions)
        log_prob = self.policy.log_prob(pred_actions)
        assert pred_actions.shape == (self.batch_size,) + self.action_space.shape

        Q_estimate = torch.min(*self.double_critic(states, pred_actions))
        assert Q_estimate.shape == (self.batch_size, 1)

        self._metrics["policy/entropy"] = -float(log_prob.detach().mean())
        loss = (self.alpha * log_prob - Q_estimate).mean()

        # Update alpha
        if self.alpha_lr is not None:
            self.alpha_optimizer.zero_grad()
            # loss_alpha = -(self.alpha * (log_prob + self.target_entropy).detach()).mean()  # CORRECT?
            loss_alpha = -(self.alpha * log_prob.detach()).mean()
            loss_alpha.backward()
            nn.utils.clip_grad_norm_(self.log_alpha, self.max_grad_norm_alpha)
            self.alpha_optimizer.step()

        self.double_critic.requires_grad_ = True
        return loss

    def learn(self, samples):
        """update the critics and actors of all the agents"""
        batch_obs_shape = (self.batch_size,) + self.obs_space.shape
        batch_action_shape = (self.batch_size,) + self.action_space.shape

        rewards = to_tensor(samples["reward"]).float().to(self.device).view(self.batch_size, 1)
        dones = to_tensor(samples["done"]).int().to(self.device).view(self.batch_size, 1)
        obss = to_tensor(samples["obs"]).float().to(self.device).view(batch_obs_shape)
        next_obss = to_tensor(samples["next_obs"]).float().to(self.device).view(batch_obs_shape)
        actions = to_tensor(samples["action"]).to(self.device).view(batch_action_shape)

        self.actor_optimizer.zero_grad()
        policy_loss = self.compute_policy_loss(obss)
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm_actor)
        self.actor_optimizer.step()

        # Critic (value) update
        c_loss = np.zeros(self.critic_number_updates)
        for idx in range(self.critic_number_updates):
            self.critic_optimizer.zero_grad()
            value_loss, error = self.compute_value_loss(obss, actions, rewards, next_obss, dones)
            value_loss.backward()
            # nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm_critic)
            self.critic_optimizer.step()
            c_loss[idx] = float(value_loss.item())
        self._loss_critic = c_loss.mean()

        # Actor (policy) update
        a_loss = np.zeros(self.actor_number_updates)
        for idx in range(self.actor_number_updates):
            self.actor_optimizer.zero_grad()
            policy_loss = self.compute_policy_loss(obss)
            policy_loss.backward()
            # nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm_actor)
            self.actor_optimizer.step()
            a_loss[idx] = float(policy_loss.item())

        self._loss_actor = a_loss.mean()

        if hasattr(self.buffer, "priority_update"):
            assert any(~torch.isnan(error))
            self.buffer.priority_update(samples["index"], error.abs())

        soft_update(self.target_double_critic, self.double_critic, self.tau)

    def log_metrics(self, data_logger: DataLogger, step: int, full_log: bool = False):
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

    def get_state(self) -> AgentState:
        return AgentState(
            model=self.model,
            obs_space=self.obs_space,
            action_space=self.action_space,
            buffer=copy.deepcopy(self.buffer.get_state()),
            network=copy.deepcopy(self.get_network_state()),
            config=self._config,
        )

    def get_network_state(self) -> NetworkState:
        return NetworkState(
            net=dict(
                policy=self.policy.state_dict(),
                actor=self.actor.state_dict(),
                double_critic=self.double_critic.state_dict(),
                target_double_critic=self.target_double_critic.state_dict(),
            )
        )

    def set_buffer(self, buffer_state: BufferState) -> None:
        self.buffer = BufferFactory.from_state(buffer_state)

    def set_network(self, network_state: NetworkState) -> None:
        self.policy.load_state_dict(network_state.net["policy"])
        self.actor.load_state_dict(network_state.net["actor"])
        self.double_critic.load_state_dict(network_state.net["double_critic"])
        self.target_double_critic.load_state_dict(network_state.net["target_double_critic"])

    @staticmethod
    def from_state(state: AgentState) -> AgentBase:
        config = copy.copy(state.config)
        config.update({"obs_space": state.obs_space, "action_space": state.action_space})
        agent = SACAgent(**config)
        if state.network is not None:
            agent.set_network(state.network)
        if state.buffer is not None:
            agent.set_buffer(state.buffer)
        return agent

    def save_state(self, path: str):
        agent_state = self.get_state()
        torch.save(agent_state, path)

    def load_state(self, path: str):
        agent_state = torch.load(path)
        self._config = agent_state.get("config", {})
        self.__dict__.update(**self._config)

        self.actor.load_state_dict(agent_state["actor"])
        self.policy.load_state_dict(agent_state["policy"])
        self.double_critic.load_state_dict(agent_state["double_critic"])
        self.target_double_critic.load_state_dict(agent_state["target_double_critic"])
