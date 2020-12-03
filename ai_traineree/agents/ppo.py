import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_traineree import DEVICE
from ai_traineree.agents.utils import compute_gae, revert_norm_returns
from ai_traineree.buffers import ReplayBuffer
from ai_traineree.networks.bodies import ActorBody, CriticBody
from ai_traineree.policies import DirichletPolicy, MultivariateGaussianPolicy
from ai_traineree.types import AgentType
from ai_traineree.utils import to_tensor
from typing import Tuple


class PPOAgent(AgentType):
    """
    Proximal Policy Optimization (PPO) [1] is an online policy gradient method
    that could be considered as an implementation-wise simplified version of
    the Trust Region Policy Optimization (TRPO).


    [1] "Proximal Policy Optimization Algorithms" (2017) by J. Schulman, F. Wolski,
        P. Dhariwal, A. Radford, O. Klimov. https://arxiv.org/abs/1707.06347
    """

    name = "PPO"

    def __init__(self, state_size: int, action_size: int, hidden_layers=(200, 200), device=None, **kwargs):
        """
        Parameters:
            is_discrete: (default: False) Whether return discrete action.
            shared_opt: (default: False) Whether a single optimier for all updates.
                In case of share optimizer, actor_lr and actor_betas are used.
            kl_div: (default: False) Whether to use KL divergence in loss.
            using_gae: (default: True) Whether to use General Advantage Estimator.
            gae_lambda: (default: 0.9) Value of \lambda in GAE.
            actor_lr: (default: 0.0003) Learning rate for the actor (policy).
            critic_lr: (default: 0.001) Learning rate for the critic (value function).
            actor_betas: (default: (0.9, 0.999) Adam's betas for actor optimizer.
            critic_betas: (default: (0.9, 0.999) Adam's betas for critic optimizer.
            gamma: (default: 0.99) Discount value.
            ppo_ratio_clip: (default: 0.02) Policy ratio clipping value.
            rollout_length: (default: 48) Number of actions to take before update.
            batch_size: (default: rollout_length) Number of samples used in learning.
            number_updates: (default: 1) How many times to learn from a rollout.
            entropy_weight: (default: 0.005) Weight of the entropy term in the loss.
            value_weight: (default: 0.005) Weight of the entropy term in the loss.

        """
        self.device = device if device is not None else DEVICE

        self.state_size = state_size
        self.action_size = action_size
        self.iteration = 0

        self.is_discrete = bool(kwargs.get("is_discrete", False))
        self.shared_opt = bool(kwargs.get("shared_opt", False))
        self.kl_div = bool(kwargs.get("kl_div", False))
        self.kl_beta = 0.1
        self.using_gae = bool(kwargs.get("using_gae", True))
        self.gae_lambda = float(kwargs.get("gae_lambda", 0.9))

        self.actor_lr = float(kwargs.get('actor_lr', 3e-4))
        self.actor_betas: Tuple[float, float] = kwargs.get('actor_betas', (0.9, 0.999))
        self.critic_lr = float(kwargs.get('critic_lr', 1e-3))
        self.critic_betas: Tuple[float, float] = kwargs.get('critic_betas', (0.9, 0.999))
        self.gamma: float = float(kwargs.get("gamma", 0.99))
        self.ppo_ratio_clip: float = float(kwargs.get("ppo_ratio_clip", 0.2))

        self.rollout_length: int = int(kwargs.get("rollout_length", 48))  # "Much less than the episode length"
        self.batch_size: int = int(kwargs.get("batch_size", self.rollout_length))
        self.number_updates: int = int(kwargs.get("number_updates", 1))
        self.entropy_weight: float = float(kwargs.get("entropy_weight", 0.0005))
        self.value_weight: float = float(kwargs.get("value_weight", 1.0))

        self.local_memory_buffer = {}
        self.memory = ReplayBuffer(batch_size=self.rollout_length, buffer_size=self.rollout_length)

        self.action_scale: float = float(kwargs.get("action_scale", 1))
        self.action_min: float = float(kwargs.get("action_min", -1))
        self.action_max: float = float(kwargs.get("action_max", 1))
        self.max_grad_norm_actor: float = float(kwargs.get("max_grad_norm_actor", 10.0))
        self.max_grad_norm_critic: float = float(kwargs.get("max_grad_norm_critic", 10.0))

        self.hidden_layers = kwargs.get('hidden_layers', hidden_layers)
        # self.policy = DirichletPolicy()  # TODO: Apparently Beta dist is better than Normal in PPO. Leaving for validation.
        # self.actor = ActorBody(state_size, self.policy.param_dim*action_size, self.hidden_layers, gate=F.relu, gate_out=None).to(self.device)
        self.policy = MultivariateGaussianPolicy(action_size, self.batch_size, device=self.device)
        self.actor = ActorBody(state_size, self.policy.param_dim*action_size, self.hidden_layers, gate=torch.tanh, gate_out=None).to(self.device)
        self.critic = CriticBody(state_size, action_size, self.hidden_layers).to(self.device)

        self.actor_params = list(self.actor.parameters())
        self.critic_params = list(self.critic.parameters())
        if self.shared_opt:
            self.opt_params = self.actor_params + self.critic_params
            self.opt = torch.optim.Adam(self.opt_params, lr=self.actor_lr, betas=self.actor_betas)
        else:
            self.actor_opt = torch.optim.Adam(self.actor_params, lr=self.actor_lr, betas=self.actor_betas)
            self.critic_opt = torch.optim.Adam(self.critic_params, lr=self.critic_lr, betas=self.critic_betas)
        self.actor_loss = 0
        self.critic_loss = 0

    def __clear_memory(self):
        self.memory = ReplayBuffer(batch_size=self.rollout_length, buffer_size=self.rollout_length)

    def act(self, state, epsilon: float=0.):
        with torch.no_grad():
            state = to_tensor(state).view(1, -1).float().to(self.device)
            actor_est = self.actor.act(state)
            assert not torch.any(torch.isnan(actor_est))

            dist = self.policy(actor_est)
            action = dist.sample()
            self.local_memory_buffer['value'] = self.critic.act(state, action)
            self.local_memory_buffer['logprob'] = self.policy.log_prob(dist, action)

            if self.is_discrete:
                # *Technically* it's the max of Softmax but that's monotonical.
                return int(torch.argmax(action))

            action = torch.clamp(action*self.action_scale, self.action_min, self.action_max)
            return action.flatten().tolist()

    def step(self, states, actions, rewards, next_state, done, **kwargs):
        self.iteration += 1
        actions = [a/self.action_scale for a in actions]

        self.memory.add(
            state=states, action=actions, reward=rewards, done=done,
            logprob=self.local_memory_buffer['logprob'], value=self.local_memory_buffer['value']
        )

        if self.iteration % self.rollout_length == 0:
            self.train()
            self.__clear_memory()

    def train(self):
        """
        Main loop that initiates the training.
        """
        experiences = self.memory.sample()
        rewards = to_tensor(experiences['reward']).to(self.device).unsqueeze(1)
        dones = to_tensor(experiences['done']).type(torch.int).to(self.device).unsqueeze(1)
        states = to_tensor(experiences['state']).float().to(self.device)
        actions = to_tensor(experiences['action']).to(self.device)
        values = torch.cat(experiences['value']).to(self.device)
        log_probs = torch.cat(experiences['logprob']).to(self.device)

        with torch.no_grad():
            if self.using_gae:
                next_val = self.critic.act(states[-1], actions[-1])
                advantages = compute_gae(rewards, dones, values, next_val, self.gamma, self.gae_lambda)
                returns = advantages + values
            else:
                values = (values - values.mean()) / values.std()
                returns = revert_norm_returns(rewards, dones, self.gamma, device=self.device).unsqueeze(1)
                advantages = returns - values

        all_indices = range(self.rollout_length)
        for _ in range(self.number_updates):
            rand_ids = random.sample(all_indices, self.batch_size)
            samples = states[rand_ids].detach(), actions[rand_ids].detach(), log_probs[rand_ids].detach(),\
                returns[rand_ids].detach(), advantages[rand_ids].detach()
            self.learn(samples)

    def learn(self, samples):
        states, actions, old_log_probs, returns, advantages = samples

        actor_est = self.actor(states.detach())
        dist = self.policy(actor_est)
        action_mu = dist.rsample()
        value = self.critic(states.detach(), action_mu.detach())

        if not self.using_gae:
            value = (value - value.mean()) / max(value.std(), 1e-8)

        entropy = dist.entropy()
        new_log_probs = self.policy.log_prob(dist, actions.detach())

        # advantages = advantages.unsqueeze(1)
        r_theta = (new_log_probs - old_log_probs).exp().unsqueeze(-1)
        r_theta_clip = torch.clamp(r_theta, 1.0 - self.ppo_ratio_clip, 1.0 + self.ppo_ratio_clip)

        if self.kl_div:
            kl_div = F.kl_div(old_log_probs.exp(), new_log_probs.exp(), reduction='mean')  # Reverse KL, see [2]
            policy_loss = -torch.mean(r_theta * advantages) + self.kl_beta * kl_div
        else:
            policy_loss = -torch.min(r_theta * advantages, r_theta_clip * advantages).mean()
        entropy_loss = -self.entropy_weight * entropy.mean()
        actor_loss = policy_loss + entropy_loss

        # Update value and critic loss
        value_loss = 0.5 * F.mse_loss(returns, value)

        if self.shared_opt:
            loss = policy_loss + self.value_weight * value_loss + entropy_loss
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.opt_params, self.max_grad_norm_actor)
            self.opt.step()
            self.actor_loss = policy_loss.mean().item()
            self.critic_loss = value_loss.mean().item()
        else:
            # Update policy and actor loss
            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm_actor)
            self.actor_opt.step()
            self.actor_loss = actor_loss.item()

            self.critic_opt.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm_critic)
            self.critic_opt.step()
            self.critic_loss = value_loss.mean().item()

    def log_writer(self, writer, episode):
        writer.add_scalar("loss/actor", self.actor_loss, episode)
        writer.add_scalar("loss/critic", self.critic_loss, episode)

    def save_state(self, path: str):
        agent_state = dict(policy=self.policy.state_dict(), actor=self.actor.state_dict(), critic=self.critic.state_dict())
        torch.save(agent_state, path)

    def load_state(self, path: str):
        agent_state = torch.load(path)
        self.policy.load_state_dict(agent_state['policy'])
        self.actor.load_state_dict(agent_state['actor'])
        self.critic.load_state_dict(agent_state['critic'])
