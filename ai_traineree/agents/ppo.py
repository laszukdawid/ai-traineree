from collections import defaultdict
from ai_traineree import DEVICE
from ai_traineree.networks import ActorBody, CriticBody
from ai_traineree.types import AgentType
from ai_traineree.policies import GaussianPolicy
import torch
import torch.nn as nn

import numpy as np

from ai_traineree.agents.utils import revert_norm_returns
from ai_traineree.buffers import ReplayBuffer


class PPOAgent(AgentType):

    name = "PPO"

    def __init__(self, state_size: int, action_size: int, hidden_layers=(300, 200), config=None, device=None, **kwargs):
        config = config if config is not None else {}
        self.device = device if device is not None else DEVICE

        self.state_size = state_size
        self.action_size = action_size
        self.iteration = 0

        self.actor_lr = float(config.get('actor_lr', 3e-4))
        self.critic_lr = float(config.get('critic_lr', 1e-3))
        self.gamma: float = float(config.get("gamma", 0.99))
        self.ppo_ratio_clip: float = float(config.get("ppo_ratio_clip", 0.2))

        self.rollout_length: int = int(config.get("rollout_length", 2048))  # "Much less than the episode length"
        self.batch_size: int = int(config.get("batch_size", self.rollout_length // 2))
        self.number_updates: int = int(config.get("number_updates", 5))
        self.entropy_weight: float = float(config.get("entropy_weight", 0.0005))
        self.value_loss_weight: float = float(config.get("value_loss_weight", 1.0))

        self.local_memory_buffer = {}
        self.memory = ReplayBuffer(batch_size=self.batch_size, buffer_size=self.rollout_length)

        self.action_scale: float = float(config.get("action_scale", 1))
        self.action_min: float = float(config.get("action_min", -2))
        self.action_max: float = float(config.get("action_max", 2))
        self.max_grad_norm_actor: float = float(config.get("max_grad_norm_actor", 100.0))
        self.max_grad_norm_critic: float = float(config.get("max_grad_norm_critic", 100.0))

        self.hidden_layers = config.get('hidden_layers', hidden_layers)
        self.actor = ActorBody(state_size, action_size, self.hidden_layers).to(self.device)
        self.critic = CriticBody(state_size, action_size, self.hidden_layers).to(self.device)
        self.policy = GaussianPolicy(action_size).to(self.device)

        self.actor_params = list(self.actor.parameters()) + [self.policy.std]
        self.critic_params = self.critic.parameters()
        self.actor_opt = torch.optim.SGD(self.actor_params, lr=self.actor_lr)
        self.critic_opt = torch.optim.SGD(self.critic_params, lr=self.critic_lr)

        self.writer = kwargs.get("writer")

    def __clear_memory(self):
        self.memory = ReplayBuffer(batch_size=self.batch_size, buffer_size=self.rollout_length)

    def act(self, state, noise=0):
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1).astype(np.float32)).to(self.device)
            action_mu = self.actor(state)
            value = self.critic(state, action_mu)

            dist = self.policy(action_mu)
            action = dist.sample()
            logprob = dist.log_prob(action)

            self.local_memory_buffer['value'] = value
            self.local_memory_buffer['logprob'] = logprob

            action = action.cpu().numpy().flatten()
            return np.clip(action*self.action_scale, self.action_min, self.action_max)

    def step(self, states, actions, rewards, next_state, done, **kwargs):
        self.iteration += 1

        self.memory.add(
            state=states, action=actions, reward=rewards, done=done,
            logprob=self.local_memory_buffer['logprob'], value=self.local_memory_buffer['value']
        )

        if self.iteration % self.rollout_length == 0:
            self.update()
            self.__clear_memory()

    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        all_indices = np.arange(self.batch_size)
        for _ in range(self.batch_size // mini_batch_size):
            rand_ids = np.random.choice(all_indices, mini_batch_size, replace=False)
            yield states[rand_ids], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], advantage[rand_ids]

    def _unpack_experiences(self, experiences):
        unpacked_experiences = defaultdict(lambda: [])
        for experience in experiences:
            unpacked_experiences['rewards'].append(experience.reward)
            unpacked_experiences['dones'].append(experience.done)
            unpacked_experiences['values'].append(experience.value)
            unpacked_experiences['states'].append(experience.state)
            unpacked_experiences['actions'].append(experience.action)
            unpacked_experiences['logprobs'].append(experience.logprob)

        return unpacked_experiences

    def update(self):
        experiences = self.memory.sample()
        rewards = torch.tensor(experiences['reward']).to(self.device)
        dones = torch.tensor(experiences['done']).type(torch.int).to(self.device)
        states = torch.tensor(experiences['state']).to(self.device)
        actions = torch.tensor(experiences['action']).to(self.device)
        values = torch.cat(experiences['value'])
        log_probs = torch.cat(experiences['logprob'])

        returns = revert_norm_returns(rewards, dones, self.gamma, device=self.device).unsqueeze(1)
        advantages = returns - values

        for _ in range(self.number_updates):
            for samples in self.ppo_iter(self.batch_size, states, actions, log_probs, returns, advantages):
                self.learn(samples)

    def learn(self, samples):
        state, action, old_log_probs, return_, advantage = samples

        action_mu = self.actor(state.detach())
        dist = self.policy(action_mu)
        value = self.critic(state.detach(), action_mu.detach())

        entropy = dist.entropy()
        new_log_probs = dist.log_prob(action.detach())

        r_theta = (new_log_probs - old_log_probs).exp()
        r_theta_clip = torch.clamp(r_theta, 1.0 - self.ppo_ratio_clip, 1.0 + self.ppo_ratio_clip)

        policy_loss = -torch.min(r_theta * advantage, r_theta_clip * advantage).mean()
        entropy_loss = -self.entropy_weight * entropy.mean()
        actor_loss = policy_loss + entropy_loss

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm_actor)
        self.actor_opt.step()
        self.actor_loss = actor_loss.item()
        # loss = policy_loss + value_loss + entropy_loss

        value_loss = self.value_loss_weight * 0.5 * (return_ - value).pow(2).mean()

        self.critic_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm_critic)
        self.critic_opt.step()
        self.critic_loss = value_loss.mean().item()

    def log_writer(self, episode):
        self.writer.add_scalar("loss/actor", self.actor_loss, episode)
        self.writer.add_scalar("loss/critic", self.critic_loss, episode)

    def save_state(self, path: str):
        agent_state = dict(policy=self.policy.state_dict())
        torch.save(agent_state, f'{path}_agent.net')

    def load_state(self, path: str):
        agent_state = torch.load(f'{path}_agent.net')
        self.policy.load_state_dict(agent_state['policy'])
