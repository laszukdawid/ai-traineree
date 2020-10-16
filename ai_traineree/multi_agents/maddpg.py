import torch
import torch.nn.functional as F
import torch.optim as optim

from ai_traineree import DEVICE
from ai_traineree.buffers import ReplayBuffer
from ai_traineree.agents.ddpg import DDPGAgent
from ai_traineree.agents.utils import hard_update, soft_update
from ai_traineree.networks import CriticBody
from ai_traineree.types import AgentType

from typing import Dict, Optional


class MADDPGAgent(AgentType):

    name = "MADDPG"

    def __init__(self, env, state_size: int, action_size: int, agents_number: int, config: Dict, **kwargs):

        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.agents_number = agents_number

        hidden_layers = config.get('hidden_layers', (256, 128))
        noise_scale = float(config.get('noise_scale', 0.2))
        noise_sigma = float(config.get('noise_sigma', 0.1))
        actor_lr = float(config.get('actor_lr', 1e-3))
        critic_lr = float(config.get('critic_lr', 1e-3))

        self.maddpg_agent = [
            DDPGAgent(
                agents_number*state_size, action_size, hidden_layers=hidden_layers,
                actor_lr=actor_lr, critic_lr=critic_lr,
                noise_scale=noise_scale, noise_sigma=noise_sigma
            ) for _ in range(agents_number)
        ]

        self.gamma: float = float(config.get('gamma', 0.99))
        self.tau: float = float(config.get('tau', 0.002))
        self.gradient_clip: Optional[float] = config.get('gradient_clip')

        self.batch_size: int = int(config.get('batch_size', 64))
        self.buffer_size = int(config.get('buffer_size', int(1e6)))
        self.buffer = ReplayBuffer(self.batch_size, self.buffer_size)

        self.warm_up: int = int(config.get('warm_up', 1e3))
        self.update_freq: int = int(config.get('update_freq', 2))
        self.number_updates: int = int(config.get('number_updates', 2))

        self.critic = CriticBody(agents_number*state_size, agents_number*action_size, hidden_layers=hidden_layers).to(DEVICE)
        self.target_critic = CriticBody(agents_number*state_size, agents_number*action_size, hidden_layers=hidden_layers).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        hard_update(self.target_critic, self.critic)

        self.reset()

    def reset(self):
        self.iteration = 0
        self.reset_agents()

    def reset_agents(self):
        for agent in self.maddpg_agent:
            agent.reset_agent()
        self.critic.reset_parameters()
        self.target_critic.reset_parameters()

    def step(self, state, action, reward, next_state, done) -> None:
        self.iteration += 1
        self.buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

        if self.iteration < self.warm_up:
            return

        if len(self.buffer) > self.batch_size and (self.iteration % self.update_freq) == 0:
            for _ in range(self.number_updates):
                for agent_number in range(self.agents_number):
                    batch = self.buffer.sample_sars()
                    self.learn(batch, agent_number)
                    # self.update_targets()

    def act(self, states, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        tensor_states = torch.tensor(states)
        with torch.no_grad():
            actions = []
            for agent in self.maddpg_agent:
                agent.actor.eval()
                actions += agent.act(tensor_states, noise)
                agent.actor.train()

        return torch.stack(actions)

    def __flatten_actions(self, actions):
        return actions.view(-1, self.agents_number*self.action_size)

    def learn(self, samples, agent_number: int) -> None:
        """update the critics and actors of all the agents """

        action_offset = agent_number*self.action_size

        # No need to flip since there are no paralle agents
        states, actions, rewards, next_states, dones = samples
        flat_states = states.view(-1, self.agents_number*self.state_size)
        flat_next_states = next_states.view(-1, self.agents_number*self.state_size)
        flat_actions = actions.view(-1, self.agents_number*self.action_size)
        agent_rewards = rewards.select(1, agent_number).view(-1, 1).detach()
        agent_dones = dones.select(1, agent_number).view(-1, 1).detach()

        agent = self.maddpg_agent[agent_number]

        next_actions = actions.detach().clone()
        next_actions.data[:, action_offset:action_offset+self.action_size] = agent.target_actor(flat_next_states)

        # critic loss
        Q_target_next = self.target_critic(flat_next_states, self.__flatten_actions(next_actions))
        Q_target = agent_rewards + (self.gamma * Q_target_next * (1 - agent_dones))
        Q_expected = self.critic(flat_states, flat_actions)
        critic_loss = F.mse_loss(Q_expected, Q_target)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()
        self.critic_loss = critic_loss.mean().item()

        # Compute actor loss
        pred_actions = actions.detach().clone()
        pred_actions.data[:, action_offset:action_offset+self.action_size] = agent.actor(flat_states)

        actor_loss = -self.critic(flat_states, self.__flatten_actions(pred_actions)).mean()
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()
        self.actor_loss = actor_loss.mean().item()

        soft_update(agent.target_actor, agent.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)

    def update_targets(self):
        """soft update targets"""
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)

    def log_writer(self, writer, episode):
        writer.add_scalar("loss/actor", self.actor_loss, episode)
        writer.add_scalar("loss/critic", self.critic_loss, episode)
