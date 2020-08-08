from ai_traineree.networks import ActorBody
from ai_traineree.types import AgentType
from ai_traineree.policies import StochasticActorCritic
import torch
import torch.nn as nn

import numpy as np

from ai_traineree.agents.utils import revert_norm_returns
from ai_traineree.buffers import ReplayBuffer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOAgent(AgentType):

    name = "PPO"

    def __init__(self, state_size: int, action_size: int, hidden_layers=(300, 200), config=None, device=None):
        """
            Expecting config to have:
            * state_dim
            * action_dim
        """
        self.config = config if config is not None else {}
        self.device = device if device is not None else DEVICE
        self.state_size = state_size
        self.action_size = action_size
        self.iteration = 0

        self.actor_lr = config.get('actor_lr', 3e-4)
        self.critic_lr = config.get('critic_lr', 1e-3)
        self.gamma = config.get("gamma", 0.99)
        self.ppo_ratio_clip = config.get("ppo_ratio_clip", 0.2)

        self.rollout_length = config.get("rollout_length", 2048)  # "Much less than the episode length"
        self.batch_size = config.get("batch_size", self.rollout_length // 2)
        self.optimization_epochs = config.get("optimization_epochs", 5)
        self.entropy_weight = config.get("entropy_weight", 0.0005)
        self.value_loss_weight = config.get("value_loss_weight", 1.0)

        self.memory = ReplayBuffer(batch_size=self.batch_size, buffer_size=self.rollout_length)

        self.action_scale = config.get("action_scale", 1)
        self.action_min = config.get("action_min", -2)
        self.action_max = config.get("action_max", 2)
        self.max_grad_norm_actor = config.get("max_grad_norm_actor", 1.0)
        self.max_grad_norm_critic = config.get("max_grad_norm_critic", 1.0)

        self.hidden_layers = config.get('hidden_layers', hidden_layers)
        actor = ActorBody(state_size, action_size, self.hidden_layers).to(self.device)
        critic = ActorBody(state_size, action_size, self.hidden_layers, gate_out=None, last_layer_range=(-1e-5, 1e-5))
        self.policy = StochasticActorCritic(state_size, action_size, self.hidden_layers, actor, critic).to(self.device)
        self.actor_opt = torch.optim.AdamW(self.policy.actor_params, lr=self.actor_lr)
        self.critic_opt = torch.optim.AdamW(self.policy.critic_params, lr=self.critic_lr)

        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_opt, step_size=1000, gamma=0.99)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_opt, step_size=1000, gamma=0.99)

        self.actor_loss = 0
        self.critic_loss = 0

    def __clear_memory(self):
        self.memory = ReplayBuffer(batch_size=self.batch_size, buffer_size=self.rollout_length)

    def act(self, state, noise=0):
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1).astype(np.float32)).to(self.device)
            dist, value = self.policy(state)
            action = dist.sample()
            logprob = dist.log_prob(action)

            self.memory.values.append(value)
            self.memory.logprobs.append(logprob)

            action = action.cpu().numpy().flatten()
            return np.clip(action*self.action_scale, self.action_min, self.action_max)

    def step(self, states, actions, rewards, next_state, done):
        self.iteration += 1

        self.memory.add_reward(torch.tensor([rewards], dtype=torch.float32).to(self.device))
        self.memory.add_done(torch.tensor([done], dtype=torch.int).to(self.device))
        self.memory.add_state(torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(self.device))
        self.memory.add_action(torch.tensor(actions, dtype=torch.float32).unsqueeze(0).to(self.device))
        # self.memory.next_states.append(next_state)

        if self.iteration % self.rollout_length == 0:
            self.update()
            self.__clear_memory()

    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        all_indices = np.arange(self.batch_size)
        for _ in range(self.batch_size // mini_batch_size):
            rand_ids = np.random.choice(all_indices, mini_batch_size, replace=False)
            yield states[rand_ids], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], advantage[rand_ids]

    def update(self):
        rewards = torch.cat(self.memory.sample_rewards()).detach()
        dones = torch.cat(self.memory.sample_dones()).detach()
        values = torch.cat(self.memory.sample_values()).detach()
        states = torch.cat(self.memory.sample_states()).detach()
        actions = torch.cat(self.memory.sample_actions()).detach()
        log_probs = torch.cat(self.memory.sample_logprobs()).detach()

        returns = revert_norm_returns(rewards, dones, self.gamma).unsqueeze(1)
        advantages = returns - values

        ppo_sampler = self.ppo_iter(self.batch_size, states, actions, log_probs, returns, advantages)
        for _ in range(self.optimization_epochs):
            for state, action, old_log_probs, return_, advantage in ppo_sampler:
                dist, value = self.policy(state.detach())
                entropy = dist.entropy()
                new_log_probs = dist.log_prob(action.detach())

                r_theta = (new_log_probs - old_log_probs).exp()
                r_theta_clip = torch.clamp(r_theta, 1.0 - self.ppo_ratio_clip, 1.0 + self.ppo_ratio_clip)

                policy_loss = -torch.min(r_theta * advantage, r_theta_clip * advantage).mean()
                entropy_loss = -self.entropy_weight * entropy.mean()
                actor_loss = policy_loss + entropy_loss

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.actor_params, self.max_grad_norm_actor)
                self.actor_opt.step()
                self.actor_loss = actor_loss.item()
                # loss = policy_loss + value_loss + entropy_loss

                value_loss = self.value_loss_weight * 0.5 * (return_ - value).pow(2).mean()

                self.critic_opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.critic_params, self.max_grad_norm_critic)
                self.critic_opt.step()
                self.critic_loss = value_loss.mean().item()

        self.actor_scheduler.step()
        self.critic_scheduler.step()
