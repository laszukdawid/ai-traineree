import copy
import itertools
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ai_traineree import DEVICE
from ai_traineree.agents import AgentBase
from ai_traineree.agents.agent_utils import compute_gae, normalize, revert_norm_returns
from ai_traineree.buffers import RolloutBuffer
from ai_traineree.buffers.buffer_factory import BufferFactory
from ai_traineree.loggers import DataLogger
from ai_traineree.networks.bodies import ActorBody
from ai_traineree.policies import MultivariateGaussianPolicy, MultivariateGaussianPolicySimple
from ai_traineree.types.state import AgentState, BufferState, NetworkState
from ai_traineree.utils import to_numbers_seq, to_tensor


class PPOAgent(AgentBase):
    """
    Proximal Policy Optimization (PPO) [1] is an online policy gradient method
    that could be considered as an implementation-wise simplified version of
    the Trust Region Policy Optimization (TRPO).


    [1] "Proximal Policy Optimization Algorithms" (2017) by J. Schulman, F. Wolski,
        P. Dhariwal, A. Radford, O. Klimov. https://arxiv.org/abs/1707.06347
    """

    name = "PPO"

    def __init__(self, state_size: int, action_size: int, **kwargs):
        """
        Parameters:
            state_size (int): Number of input dimensions.
            action_size (int): Number of output dimensions

        Keyword parameters:
            hidden_layers (tuple of ints): Tuple defining hidden dimensions in fully connected nets. Default: (100, 100).
            is_discrete (bool): Whether return discrete action. Default: False.
            kl_div (bool): Whether to use KL divergence in loss. Default: False.
            using_gae (bool): Whether to use General Advantage Estimator. Default: True.
            gae_lambda (float): Value of lambda in GAE. Default: 0.96.
            actor_lr (float): Learning rate for the actor (policy). Default: 0.0003.
            critic_lr (float): Learning rate for the critic (value function). Default: 0.001.
            actor_betas: (tuple of two floats) Adam's betas for actor optimizer. Default: (0.9, 0.999).
            critic_betas: (tuple of two floats) Adam's betas for critic optimizer. Default: (0.9, 0.999).
            gamma (float): Discount value. Default: 0.99.
            ppo_ratio_clip (float): Policy ratio clipping value. Default: 0.25.
            num_epochs (int): Number of time to learn from samples. Default: 1.
            rollout_length (int): Number of actions to take before update. Default: 48.
            batch_size (int): Number of samples used in learning. Default: `rollout_length`.
            actor_number_updates (int): Number of times policy losses are propagated. Default: 10.
            critic_number_updates (int): Number of times value losses are propagated. Default: 10.
            entropy_weight (float): Weight of the entropy term in the loss. Default: 0.005.
            value_loss_weight (float): Weight of the entropy term in the loss. Default: 0.005.
            max_grad_norm_actor (float) Maximum norm value for actor gradient. Default: 100.
            max_grad_norm_critic (float): Maximum norm value for critic gradient. Default: 100.

        """
        super().__init__(**kwargs)

        self.device = self._register_param(kwargs, "device", DEVICE)  # Default device is CUDA if available

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = to_numbers_seq(self._register_param(kwargs, "hidden_layers", (100, 100)))
        self.iteration = 0

        self.is_discrete = bool(self._register_param(kwargs, "is_discrete", False))
        self.using_gae = bool(self._register_param(kwargs, "using_gae", True))
        self.gae_lambda = float(self._register_param(kwargs, "gae_lambda", 0.96))

        self.actor_lr = float(self._register_param(kwargs, 'actor_lr', 3e-4))
        self.actor_betas: Tuple[float, float] = to_numbers_seq(self._register_param(kwargs, 'actor_betas', (0.9, 0.999)))
        self.critic_lr = float(self._register_param(kwargs, 'critic_lr', 1e-3))
        self.critic_betas: Tuple[float, float] = to_numbers_seq(self._register_param(kwargs, 'critic_betas', (0.9, 0.999)))
        self.gamma = float(self._register_param(kwargs, "gamma", 0.99))
        self.ppo_ratio_clip = float(self._register_param(kwargs, "ppo_ratio_clip", 0.25))

        self.using_kl_div = bool(self._register_param(kwargs, "using_kl_div", False))
        self.kl_beta = float(self._register_param(kwargs, 'kl_beta', 0.1))
        self.target_kl = float(self._register_param(kwargs, "target_kl", 0.01))
        self.kl_div = float('inf')

        self.num_workers = int(self._register_param(kwargs, "num_workers", 1))
        self.num_epochs = int(self._register_param(kwargs, "num_epochs", 1))
        self.rollout_length = int(self._register_param(kwargs, "rollout_length", 48))  # "Much less than the episode length"
        self.batch_size = int(self._register_param(kwargs, "batch_size", self.rollout_length))
        self.actor_number_updates = int(self._register_param(kwargs, "actor_number_updates", 10))
        self.critic_number_updates = int(self._register_param(kwargs, "critic_number_updates", 10))
        self.entropy_weight = float(self._register_param(kwargs, "entropy_weight", 0.5))
        self.value_loss_weight = float(self._register_param(kwargs, "value_loss_weight", 1.0))

        self.local_memory_buffer = {}

        self.action_scale = float(self._register_param(kwargs, "action_scale", 1))
        self.action_min = float(self._register_param(kwargs, "action_min", -1))
        self.action_max = float(self._register_param(kwargs, "action_max", 1))
        self.max_grad_norm_actor = float(self._register_param(kwargs, "max_grad_norm_actor", 100.0))
        self.max_grad_norm_critic = float(self._register_param(kwargs, "max_grad_norm_critic", 100.0))

        if kwargs.get("simple_policy", False):
            self.policy = MultivariateGaussianPolicySimple(self.action_size, **kwargs)
        else:
            self.policy = MultivariateGaussianPolicy(self.action_size, device=self.device)

        self.buffer = RolloutBuffer(batch_size=self.batch_size, buffer_size=self.rollout_length)
        self.actor = ActorBody(
            state_size, self.policy.param_dim*action_size,
            gate_out=torch.tanh, hidden_layers=self.hidden_layers, device=self.device)
        self.critic = ActorBody(
            state_size, 1,
            gate_out=None, hidden_layers=self.hidden_layers, device=self.device)
        self.actor_params = list(self.actor.parameters()) + list(self.policy.parameters())
        self.critic_params = list(self.critic.parameters())

        self.actor_opt = optim.Adam(self.actor_params, lr=self.actor_lr, betas=self.actor_betas)
        self.critic_opt = optim.Adam(self.critic_params, lr=self.critic_lr, betas=self.critic_betas)
        self._loss_actor = float('nan')
        self._loss_critic = float('nan')
        self._metrics: Dict[str, float] = {}

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
            and self._config == o._config \
            and self.buffer == o.buffer \
            and self.get_network_state() == o.get_network_state()  # TODO @dawid: Currently net isn't compared properly

    def __clear_memory(self):
        self.buffer.clear()

    @torch.no_grad()
    def act(self, state, epsilon: float=0.):
        actions = []
        logprobs = []
        values = []
        state = to_tensor(state).view(self.num_workers, self.state_size).float().to(self.device)
        for worker in range(self.num_workers):
            actor_est = self.actor.act(state[worker].unsqueeze(0))
            assert not torch.any(torch.isnan(actor_est))

            dist = self.policy(actor_est)
            action = dist.sample()
            value = self.critic.act(state[worker].unsqueeze(0))  # Shape: (1, 1)
            logprob = self.policy.log_prob(dist, action)  # Shape: (1,)
            values.append(value)
            logprobs.append(logprob)

            if self.is_discrete:  # *Technically* it's the max of Softmax but that's monotonic.
                action = int(torch.argmax(action))
            else:
                action = torch.clamp(action*self.action_scale, self.action_min, self.action_max)
                action = action.cpu().numpy().flatten().tolist()
            actions.append(action)

        self.local_memory_buffer['value'] = torch.cat(values)
        self.local_memory_buffer['logprob'] = torch.stack(logprobs)
        assert len(actions) == self.num_workers
        return actions if self.num_workers > 1 else actions[0]

    def step(self, state, action, reward, next_state, done, **kwargs):
        self.iteration += 1

        self.buffer.add(
            state=torch.tensor(state).reshape(self.num_workers, self.state_size).float(),
            action=torch.tensor(action).reshape(self.num_workers, self.action_size).float(),
            reward=torch.tensor(reward).reshape(self.num_workers, 1),
            done=torch.tensor(done).reshape(self.num_workers, 1),
            logprob=self.local_memory_buffer['logprob'].reshape(self.num_workers, 1),
            value=self.local_memory_buffer['value'].reshape(self.num_workers, 1),
        )

        if self.iteration % self.rollout_length == 0:
            self.train()
            self.__clear_memory()

    def train(self):
        """
        Main loop that initiates the training.
        """
        experiences = self.buffer.all_samples()
        rewards = to_tensor(experiences['reward']).to(self.device)
        dones = to_tensor(experiences['done']).type(torch.int).to(self.device)
        states = to_tensor(experiences['state']).to(self.device)
        actions = to_tensor(experiences['action']).to(self.device)
        values = to_tensor(experiences['value']).to(self.device)
        logprobs = to_tensor(experiences['logprob']).to(self.device)
        assert rewards.shape == dones.shape == values.shape == logprobs.shape
        assert states.shape == (self.rollout_length, self.num_workers, self.state_size), f"Wrong states shape: {states.shape}"
        assert actions.shape == (self.rollout_length, self.num_workers, self.action_size), f"Wrong action shape: {actions.shape}"

        with torch.no_grad():
            if self.using_gae:
                next_value = self.critic.act(states[-1])
                advantages = compute_gae(rewards, dones, values, next_value, self.gamma, self.gae_lambda)
                advantages = normalize(advantages)
                returns = advantages + values
                # returns = normalize(advantages + values)
                assert advantages.shape == returns.shape == values.shape
            else:
                returns = revert_norm_returns(rewards, dones, self.gamma)
                returns = returns.float()
                advantages = normalize(returns - values)
                assert advantages.shape == returns.shape == values.shape

        for _ in range(self.num_epochs):
            idx = 0
            self.kl_div = 0
            while idx < self.rollout_length:
                _states = states[idx:idx+self.batch_size].view(-1, self.state_size).detach()
                _actions = actions[idx:idx+self.batch_size].view(-1, self.action_size).detach()
                _logprobs = logprobs[idx:idx+self.batch_size].view(-1, 1).detach()
                _returns = returns[idx:idx+self.batch_size].view(-1, 1).detach()
                _advantages = advantages[idx:idx+self.batch_size].view(-1, 1).detach()
                idx += self.batch_size
                self.learn((_states, _actions, _logprobs, _returns, _advantages))

            self.kl_div = abs(self.kl_div) / (self.actor_number_updates * self.rollout_length / self.batch_size)
            if self.kl_div > self.target_kl * 1.75:
                self.kl_beta = min(2 * self.kl_beta, 1e2)  # Max 100
            if self.kl_div < self.target_kl / 1.75:
                self.kl_beta = max(0.5 * self.kl_beta, 1e-6)  # Min 0.000001
            self._metrics['policy/kl_beta'] = self.kl_beta

    def compute_policy_loss(self, samples):
        states, actions, old_log_probs, _, advantages = samples

        actor_est = self.actor(states)
        dist = self.policy(actor_est)

        entropy = dist.entropy()
        new_log_probs = self.policy.log_prob(dist, actions).view(-1, 1)
        assert new_log_probs.shape == old_log_probs.shape

        r_theta = (new_log_probs - old_log_probs).exp()
        r_theta_clip = torch.clamp(r_theta, 1.0 - self.ppo_ratio_clip, 1.0 + self.ppo_ratio_clip)
        assert r_theta.shape == r_theta_clip.shape

        # KL = E[log(P/Q)] = sum_{P}( P * log(P/Q) ) -- \approx --> avg_{P}( log(P) - log(Q) )
        approx_kl_div = (old_log_probs - new_log_probs).mean().item()
        if self.using_kl_div:
            # Ratio threshold for updates is 1.75 (although it should be configurable)
            policy_loss = -torch.mean(r_theta * advantages) + self.kl_beta * approx_kl_div
        else:
            joint_theta_adv = torch.stack((r_theta * advantages, r_theta_clip * advantages))
            assert joint_theta_adv.shape[0] == 2
            policy_loss = -torch.amin(joint_theta_adv, dim=0).mean()
        entropy_loss = -self.entropy_weight * entropy.mean()

        loss = policy_loss + entropy_loss
        self._metrics['policy/kl_div'] = approx_kl_div
        self._metrics['policy/policy_ratio'] = float(r_theta.mean())
        self._metrics['policy/policy_ratio_clip_mean'] = float(r_theta_clip.mean())
        return loss, approx_kl_div

    def compute_value_loss(self, samples):
        states, _, _, returns, _ = samples
        values = self.critic(states)
        self._metrics['value/value_mean'] = values.mean()
        self._metrics['value/value_std'] = values.std()
        return F.mse_loss(values, returns)

    def learn(self, samples):
        self._loss_actor = 0.

        for _ in range(self.actor_number_updates):
            self.actor_opt.zero_grad()
            loss_actor, kl_div = self.compute_policy_loss(samples)
            self.kl_div += kl_div
            if kl_div > 1.5 * self.target_kl:
                # Early break
                # print(f"Iter: {i:02} Early break")
                break
            loss_actor.backward()
            nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm_actor)
            self.actor_opt.step()
            self._loss_actor = loss_actor.item()

        for _ in range(self.critic_number_updates):
            self.critic_opt.zero_grad()
            loss_critic = self.compute_value_loss(samples)
            loss_critic.backward()
            nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm_critic)
            self.critic_opt.step()
            self._loss_critic = float(loss_critic.item())

    def log_metrics(self, data_logger: DataLogger, step: int, full_log: bool=False):
        data_logger.log_value("loss/actor", self._loss_actor, step)
        data_logger.log_value("loss/critic", self._loss_critic, step)
        for metric_name, metric_value in self._metrics.items():
            data_logger.log_value(metric_name, metric_value, step)

        policy_params = {str(i): v for i, v in enumerate(itertools.chain.from_iterable(self.policy.parameters()))}
        data_logger.log_values_dict("policy/param", policy_params, step)

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
            model=self.name,
            state_space=self.state_size,
            action_space=self.action_size,
            config=self._config,
            buffer=copy.deepcopy(self.buffer.get_state()),
            network=copy.deepcopy(self.get_network_state())
        )

    def get_network_state(self) -> NetworkState:
        return NetworkState(net=dict(
            policy=self.policy.state_dict(),
            actor=self.actor.state_dict(),
            critic=self.critic.state_dict(),
        ))

    def set_buffer(self, buffer_state: BufferState) -> None:
        self.buffer = BufferFactory.from_state(buffer_state)

    def set_network(self, network_state: NetworkState) -> None:
        self.policy.load_state_dict(network_state.net['policy'])
        self.actor.load_state_dict(network_state.net['actor'])
        self.critic.load_state_dict(network_state.net['critic'])

    @staticmethod
    def from_state(state: AgentState) -> AgentBase:
        config = copy.copy(state.config)
        config.update({'state_size': state.state_space, 'action_size': state.action_space})
        agent = PPOAgent(**config)
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
        self._config = agent_state.get('config', {})
        self.__dict__.update(**self._config)

        self.policy.load_state_dict(agent_state['policy'])
        self.actor.load_state_dict(agent_state['actor'])
        self.critic.load_state_dict(agent_state['critic'])
