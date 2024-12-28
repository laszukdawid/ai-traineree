import copy
import itertools
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ai_traineree import DEVICE
from ai_traineree.agents import AgentBase
from ai_traineree.agents.agent_utils import compute_gae, normalize, revert_norm_returns
from ai_traineree.buffers.buffer_factory import BufferFactory
from ai_traineree.buffers.rollout import RolloutBuffer
from ai_traineree.loggers import DataLogger
from ai_traineree.networks.bodies import ActorBody
from ai_traineree.policies import MultivariateGaussianPolicy, MultivariateGaussianPolicySimple
from ai_traineree.types import ActionType, AgentState, BufferState, NetworkState
from ai_traineree.types.dataspace import DataSpace
from ai_traineree.types.experience import Experience
from ai_traineree.utils import to_numbers_seq, to_tensor


class PPOAgent(AgentBase):
    """
    Proximal Policy Optimization (PPO) [1] is an online policy gradient method
    that could be considered as an implementation-wise simplified version of
    the Trust Region Policy Optimization (TRPO).


    [1] "Proximal Policy Optimization Algorithms" (2017) by J. Schulman, F. Wolski,
        P. Dhariwal, A. Radford, O. Klimov. https://arxiv.org/abs/1707.06347
    """

    model = "PPO"
    logger = logging.getLogger("PPO")

    def __init__(self, obs_space: DataSpace, action_space: DataSpace, **kwargs):
        """
        Parameters:
            obs_space (DataSpace): Dataspace describing the input.
            action_space (DataSpace): Dataspace describing the output.

        Keyword arguments:
            hidden_layers (tuple of ints): Shape of the hidden layers in fully connected network. Default: (128, 128).
            is_discrete (bool): Whether return discrete action. Default: False.
            using_kl_div (bool): Whether to use KL divergence in loss. Default: False.
            using_gae (bool): Whether to use General Advantage Estimator. Default: True.
            gae_lambda (float): Value of lambda in GAE. Default: 0.96.
            actor_lr (float): Learning rate for the actor (policy). Default: 0.0003.
            critic_lr (float): Learning rate for the critic (value function). Default: 0.001.
            gamma (float): Discount value. Default: 0.99.
            ppo_ratio_clip (float): Policy ratio clipping value. Default: 0.25.
            num_epochs (int): Number of time to learn from samples. Default: 1.
            rollout_length (int): Number of actions to take before update. Default: 48.
            batch_size (int): Number of samples used in learning. Default: `rollout_length`.
            actor_number_updates (int): Number of times policy losses are propagated. Default: 10.
            critic_number_updates (int): Number of times value losses are propagated. Default: 10.
            entropy_weight (float): Weight of the entropy term in the loss. Default: 0.005.
            max_grad_norm_actor (float) Maximum norm value for actor gradient. Default: 100.
            max_grad_norm_critic (float): Maximum norm value for critic gradient. Default: 100.

        """
        super().__init__(**kwargs)

        self.device = self._register_param(kwargs, "device", DEVICE, update=True)  # Default device is CUDA if available

        self.obs_space = obs_space
        self.action_space = action_space
        assert len(action_space.shape) == 1, "Only 1D actions are supported"
        self.action_size = action_space.shape[0]

        self._config["obs_space"] = self.obs_space
        self._config["action_space"] = self.action_space
        self.hidden_layers = to_numbers_seq(self._register_param(kwargs, "hidden_layers", (128, 128)))
        self.iteration = 0

        self.is_discrete = bool(self._register_param(kwargs, "is_discrete", False))
        self.using_gae = bool(self._register_param(kwargs, "using_gae", True))
        self.gae_lambda = float(self._register_param(kwargs, "gae_lambda", 0.96))

        self.actor_lr = float(self._register_param(kwargs, "actor_lr", 3e-4))
        self.critic_lr = float(self._register_param(kwargs, "critic_lr", 1e-3))
        self.gamma = float(self._register_param(kwargs, "gamma", 0.99))
        self.ppo_ratio_clip = float(self._register_param(kwargs, "ppo_ratio_clip", 0.25))

        self.using_kl_div = bool(self._register_param(kwargs, "using_kl_div", False))
        self.kl_beta = float(self._register_param(kwargs, "kl_beta", 0.1))
        self.target_kl = float(self._register_param(kwargs, "target_kl", 0.01))
        self.kl_div = float("inf")

        self.num_workers = int(self._register_param(kwargs, "num_workers", 1))
        self.num_epochs = int(self._register_param(kwargs, "num_epochs", 1))
        self.rollout_length = int(self._register_param(kwargs, "rollout_length", 48))  # "Much shorter than episode"
        self.batch_size = int(self._register_param(kwargs, "batch_size", self.rollout_length))
        self.actor_number_updates = int(self._register_param(kwargs, "actor_number_updates", 10))
        self.critic_number_updates = int(self._register_param(kwargs, "critic_number_updates", 10))
        self.entropy_loss_weight = float(self._register_param(kwargs, "entropy_loss_weight", 0.5))

        self.max_grad_norm_actor = float(self._register_param(kwargs, "max_grad_norm_actor", 100.0))
        self.max_grad_norm_critic = float(self._register_param(kwargs, "max_grad_norm_critic", 100.0))

        if kwargs.get("simple_policy", False):
            self.policy = MultivariateGaussianPolicySimple(self.action_size, **kwargs)
        else:
            self.policy = MultivariateGaussianPolicy(self.action_size, device=self.device)

        self.buffer = RolloutBuffer(batch_size=self.batch_size, buffer_size=self.rollout_length)
        self.actor = ActorBody(
            self.obs_space.shape,
            (self.policy.param_dim * self.action_size,),
            gate_out=torch.tanh,
            hidden_layers=self.hidden_layers,
            device=self.device,
        )
        self.critic = ActorBody(
            self.obs_space.shape, (1,), gate_out=nn.Identity(), hidden_layers=self.hidden_layers, device=self.device
        )
        self.actor_params = list(self.actor.parameters()) + list(self.policy.parameters())
        self.critic_params = list(self.critic.parameters())

        self.actor_opt = optim.Adam(self.actor_params, lr=self.actor_lr)
        self.critic_opt = optim.Adam(self.critic_params, lr=self.critic_lr)
        self._loss_actor = float("nan")
        self._loss_critic = float("nan")
        self._metrics: dict[str, float] = {}

    @property
    def loss(self) -> dict[str, float]:
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

    def __clear_memory(self):
        self.buffer.clear()

    @torch.no_grad()
    def act(self, experience: Experience, noise: float = 0.0) -> Experience:
        """Acting on the observations. Returns action.

        Parameters:
            experience (Experience): current state
            noise (float): epsilon, for epsilon-greedy action selection

        Returns:
            Experience updated with action taken.

        """
        actions: list[ActionType] = []
        logprobs = []
        values = []
        t_obs = to_tensor(experience.obs).view((self.num_workers,) + self.obs_space.shape).float().to(self.device)
        for worker in range(self.num_workers):
            actor_est = self.actor.act(t_obs[worker].unsqueeze(0))
            assert not torch.any(torch.isnan(actor_est))

            action = self.policy(actor_est)
            value = self.critic.act(t_obs[worker].unsqueeze(0))  # Shape: (1, 1)
            logprob = self.policy.log_prob(action)  # Shape: (1,)
            values.append(value)
            logprobs.append(logprob)

            if self.is_discrete:  # *Technically* it's the max of Softmax but that's monotonic.
                action = int(torch.argmax(action))
            else:
                action = action.cpu().numpy().flatten().tolist()
            actions.append(action)

        value = torch.cat(values)
        logprob = torch.stack(logprobs)
        action = actions if self.num_workers > 1 else actions[0]
        experience.update(action=action, value=torch.cat(values), logprob=torch.stack(logprobs))
        return experience

    def step(self, experience: Experience) -> None:
        """Step agent's internal learning mechanisms.

        Updates buffer with currenct experience and increments learning counter.
        When the learning counter hits `rollout_length` when we commence learning session.
        The learning counter isn't updated when the agent is in `test` mode.

        """

        if not self.train:
            return

        self.iteration += 1

        self.buffer.add(
            obs=torch.tensor(experience.obs).reshape((self.num_workers,) + self.obs_space.shape).float(),
            action=torch.tensor(experience.action).reshape((self.num_workers,) + self.action_space.shape).float(),
            reward=torch.tensor(experience.reward).reshape(self.num_workers, 1),
            done=torch.tensor(experience.done).reshape(self.num_workers, 1),
            logprob=experience.get("logprob").reshape(self.num_workers, 1),
            value=experience.get("value").reshape(self.num_workers, 1),
        )

        if self.iteration % self.rollout_length == 0:
            self.train_agent()
            self.__clear_memory()

    def train_agent(self):
        """
        Main loop that initiates the training.
        """
        experiences = self.buffer.all_samples()
        rewards = to_tensor(experiences["reward"]).to(self.device)
        dones = to_tensor(experiences["done"]).type(torch.int).to(self.device)
        obss = to_tensor(experiences["obs"]).to(self.device)
        actions = to_tensor(experiences["action"]).to(self.device)
        values = to_tensor(experiences["value"]).to(self.device)
        logprobs = to_tensor(experiences["logprob"]).to(self.device)
        assert rewards.shape == dones.shape == values.shape == logprobs.shape
        assert (
            obss.shape == (self.rollout_length, self.num_workers) + self.obs_space.shape
        ), f"Wrong obss shape: {obss.shape}"
        assert (
            actions.shape == (self.rollout_length, self.num_workers) + self.action_space.shape
        ), f"Wrong action shape: {actions.shape}"

        with torch.no_grad():
            if self.using_gae:
                next_value = self.critic.act(obss[-1])
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
                _states = obss[idx : idx + self.batch_size].view((-1,) + self.obs_space.shape).detach()
                _actions = actions[idx : idx + self.batch_size].view((-1,) + self.action_space.shape).detach()
                _logprobs = logprobs[idx : idx + self.batch_size].view(-1, 1).detach()
                _returns = returns[idx : idx + self.batch_size].view(-1, 1).detach()
                _advantages = advantages[idx : idx + self.batch_size].view(-1, 1).detach()
                idx += self.batch_size
                self.learn((_states, _actions, _logprobs, _returns, _advantages))

            self.kl_div = abs(self.kl_div) / (
                self.actor_number_updates * self.num_workers * self.rollout_length / self.batch_size
            )

            if self.using_kl_div:
                if self.kl_div > self.target_kl * 1.5:
                    self.kl_beta = min(1.5 * self.kl_beta, 1e2)  # Max 100
                elif self.kl_div < self.target_kl / 1.5:
                    self.kl_beta = max(0.75 * self.kl_beta, 1e-6)  # Min 0.000001

            if self.kl_div > self.target_kl * 1.5:
                self.logger.warning("Early stopping")
                break
            self._metrics["policy/kl_beta"] = self.kl_beta

    def compute_policy_loss(self, samples):
        obss, actions, old_log_probs, _, advantages = samples

        actor_est = self.actor(obss)
        _ = self.policy(actor_est)

        dist = self.policy._last_dist
        entropy = dist.entropy().reshape(actor_est.shape[:-1] + (1,))
        new_log_probs = self.policy.log_prob(actions).reshape(old_log_probs.shape)
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
        entropy_loss = -self.entropy_loss_weight * entropy.mean()

        loss = policy_loss + entropy_loss
        self._metrics["policy/kl_div"] = approx_kl_div
        self._metrics["policy/policy_ratio"] = float(r_theta.mean())
        self._metrics["policy/policy_ratio_clip_mean"] = float(r_theta_clip.mean())
        return loss, approx_kl_div

    def compute_value_loss(self, samples):
        obss, _, _, returns, _ = samples
        values = self.critic(obss)
        self._metrics["value/value_mean"] = values.mean()
        self._metrics["value/value_std"] = values.std()
        return F.mse_loss(values, returns)

    def learn(self, samples):
        self._loss_actor = 0.0

        for actor_iter in range(self.actor_number_updates):
            self.actor_opt.zero_grad()
            loss_actor, kl_div = self.compute_policy_loss(samples)
            self.kl_div += kl_div
            if kl_div > 1.5 * self.target_kl:
                # Early break
                self.logger.warning(
                    "Early break after %i iterations. %f > %f", actor_iter, kl_div, 1.5 * self.target_kl
                )
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

    def log_metrics(self, data_logger: DataLogger, step: int, full_log: bool = False):
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
            model=self.model,
            obs_space=self.obs_space,
            action_space=self.action_space,
            config=self._config,
            buffer=copy.deepcopy(self.buffer.get_state()),
            network=copy.deepcopy(self.get_network_state()),
        )

    def get_network_state(self) -> NetworkState:
        return NetworkState(
            net=dict(
                policy=self.policy.state_dict(),
                actor=self.actor.state_dict(),
                critic=self.critic.state_dict(),
            )
        )

    def set_buffer(self, buffer_state: BufferState) -> None:
        self.buffer = BufferFactory.from_state(buffer_state)

    def set_network(self, network_state: NetworkState) -> None:
        self.policy.load_state_dict(network_state.net["policy"])
        self.actor.load_state_dict(network_state.net["actor"])
        self.critic.load_state_dict(network_state.net["critic"])

    @staticmethod
    def from_state(state: AgentState) -> AgentBase:
        config = copy.copy(state.config)
        config.update({"obs_space": state.obs_space, "action_space": state.action_space})
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
        self._config = agent_state.get("config", {})
        self.__dict__.update(**self._config)

        self.policy.load_state_dict(agent_state["policy"])
        self.actor.load_state_dict(agent_state["actor"])
        self.critic.load_state_dict(agent_state["critic"])
