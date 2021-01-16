import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ai_traineree import DEVICE
from ai_traineree.buffers import ReplayBuffer
from ai_traineree.agents.ddpg import DDPGAgent
from ai_traineree.agents.utils import hard_update, soft_update
from ai_traineree.networks.bodies import CriticBody
from ai_traineree.types import ActionType, MultiAgentType, StateType
from ai_traineree.utils import to_tensor

from typing import Dict, List, Optional


class MADDPGAgent(MultiAgentType):

    name = "MADDPG"

    def __init__(self, state_size: int, action_size: int, agents_number: int, **kwargs):
        """Initiation of the Multi Agent DDPG.

        All keywords are also passed to DDPG agents.

        Parameters:
            state_size (int): Dimensionality of the state.
            action_size (int): Dimensionality of the action.
            agents_number (int): Number of agents.
        
        Keyword parameters:
            hidden_layers (tuple of ints): Shape for fully connected hidden layers.
            noise_scale (float): Default: 1.0. Noise amplitude.
            noise_sigma (float): Default: 0.5. Noise variance.
            actor_lr (float): Default: 0.001. Learning rate for actor network.
            critic_lr (float): Default: 0.001. Learning rate for critic network.
            gamma (float): Default: 0.99. Discount value
            tau (float): Default: 0.02. Soft copy value.
            gradient_clip (optional float): Max norm for learning gradient. If None then no clip.
            batch_size (int): Number of samples per learning.
            buffer_size (int): Number of previous samples to remember.
            warm_up (int): Number of samples to see before start learning.
            update_freq (int): How many samples between learning sessions.
            number_updates (int): How many learning cycles per learning session.

        """

        self.device = self._register_param(kwargs, "device", DEVICE)
        self.state_size: int = state_size
        self.action_size = action_size
        self.agents_number = agents_number

        hidden_layers = self._register_param(kwargs, 'hidden_layers', (256, 128))
        noise_scale = float(self._register_param(kwargs, 'noise_scale', 0.5))
        noise_sigma = float(self._register_param(kwargs, 'noise_sigma', 1.0))
        actor_lr = float(self._register_param(kwargs, 'actor_lr', 1e-3))
        critic_lr = float(self._register_param(kwargs, 'critic_lr', 1e-3))

        self.agents: List[DDPGAgent] = [
            DDPGAgent(
                agents_number*state_size, action_size, hidden_layers=hidden_layers,
                actor_lr=actor_lr, critic_lr=critic_lr,
                noise_scale=noise_scale, noise_sigma=noise_sigma,
                device=self.device,
                **kwargs,
            ) for _ in range(agents_number)
        ]

        self.gamma: float = float(self._register_param(kwargs, 'gamma', 0.99))
        self.tau: float = float(self._register_param(kwargs, 'tau', 0.002))
        self.gradient_clip: Optional[float] = self._register_param(kwargs, 'gradient_clip')

        self.batch_size: int = int(self._register_param(kwargs, 'batch_size', 64))
        self.buffer_size = int(self._register_param(kwargs, 'buffer_size', int(1e6)))
        self.buffer = ReplayBuffer(self.batch_size, self.buffer_size)

        self.warm_up: int = int(self._register_param(kwargs, 'warm_up', 0))
        self.update_freq: int = int(self._register_param(kwargs, 'update_freq', 1))
        self.number_updates: int = int(self._register_param(kwargs, 'number_updates', 1))

        self.critic = CriticBody(agents_number*state_size, agents_number*action_size, hidden_layers=hidden_layers).to(self.device)
        self.target_critic = CriticBody(agents_number*state_size, agents_number*action_size, hidden_layers=hidden_layers).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        hard_update(self.target_critic, self.critic)

        self._loss_actor: float = 0.
        self._loss_critic: float = 0.
        self.reset()

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

    def reset(self):
        self.iteration = 0
        self.reset_agents()

    def reset_agents(self):
        for agent in self.agents:
            agent.reset_agent()
        self.critic.reset_parameters()
        self.target_critic.reset_parameters()

    def step(self, states: List[StateType], actions: List[ActionType], rewards, next_states, dones) -> None:
        self.iteration += 1
        self.buffer.add(state=states, action=actions, reward=rewards, next_state=next_states, done=dones)

        if self.iteration < self.warm_up:
            return

        if len(self.buffer) > self.batch_size and (self.iteration % self.update_freq) == 0:
            for _ in range(self.number_updates):
                for agent_number in range(self.agents_number):
                    self.learn(self.buffer.sample(), agent_number)
            self.update_targets()

    def act(self, states: List[StateType], noise: float=0.0) -> List[ActionType]:
        """Get actions from all agents. Synchronized action.

        Parameters:
            states: List of states per agent. Positions need to be consistent.
            noise: Scale for the noise to include

        Returns:
            actions: List of actions that each agent wants to perform

        """
        tensor_states = torch.tensor(states).reshape(1, -1)
        with torch.no_grad():
            actions = []
            for agent in self.agents:
                agent.actor.eval()
                actions.append(agent.act(tensor_states, noise))
                agent.actor.train()

        # return torch.stack(actions)
        return actions

    def __flatten_actions(self, actions):
        return actions.view(-1, self.agents_number*self.action_size)

    def learn(self, experiences, agent_number: int) -> None:
        """update the critics and actors of all the agents """

        # TODO: Just look at this mess.
        agent_rewards = to_tensor(experiences['reward']).select(1, agent_number).float().to(self.device).unsqueeze(1).detach()
        agent_dones = to_tensor(experiences['done']).select(1, agent_number).type(torch.int).to(self.device).unsqueeze(1)
        states = to_tensor(experiences['state']).float().to(self.device).squeeze(2)
        actions = to_tensor(experiences['action']).to(self.device).squeeze(2)
        next_states = to_tensor(experiences['next_state']).float().to(self.device).squeeze(2)
        flat_states = states.view(-1, self.agents_number*self.state_size)
        flat_next_states = next_states.view(-1, self.agents_number*self.state_size)
        flat_actions = actions.view(-1, self.agents_number*self.action_size)

        agent = self.agents[agent_number]

        next_actions = actions.detach().clone()
        next_actions.data[:, agent_number] = agent.target_actor(flat_next_states)

        # critic loss
        Q_target_next = self.target_critic(flat_next_states, self.__flatten_actions(next_actions))
        Q_target = agent_rewards + (self.gamma * Q_target_next * (1 - agent_dones))
        Q_expected = self.critic(flat_states, flat_actions)
        loss_critic = F.mse_loss(Q_expected, Q_target)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        if self.gradient_clip:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()
        self._loss_critic = float(loss_critic.mean().item())

        # Compute actor loss
        pred_actions = actions.detach().clone()
        pred_actions.data[:, agent_number] = agent.actor(flat_states)

        loss_actor = -self.critic(flat_states, self.__flatten_actions(pred_actions)).mean()
        agent.actor_optimizer.zero_grad()
        loss_actor.backward()
        agent.actor_optimizer.step()
        self._loss_actor = loss_actor.mean().item()

    def update_targets(self):
        """soft update targets"""
        for ddpg_agent in self.agents:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)

    def log_metrics(self, writer, episode):
        writer.add_scalar("loss/actor", self._loss_actor, episode)
        writer.add_scalar("loss/critic", self._loss_critic, episode)

    def save_state(self, path: str):
        agents_state = {}
        agents_state['config'] = self._config
        for agent_id, agent in enumerate(self.agents):
            agents_state[f'actor_{agent_id}'] = agent.actor.state_dict()
            agents_state[f'target_actor_{agent_id}'] = agent.target_actor.state_dict()
            agents_state[f'critic_{agent_id}'] = agent.critic.state_dict()
            agents_state[f'target_critic_{agent_id}'] = agent.target_critic.state_dict()
            agents_state[f'config_{agent_id}'] = agent._config
        torch.save(agents_state, path)

    def load_state(self, path: str):
        agent_state = torch.load(path)
        self._config = agent_state.get('config', {})
        self.__dict__.update(**self._config)
        for agent_id, agent in enumerate(self.agents):
            agent.actor.load_state_dict(agent_state[f'actor_{agent_id}'])
            agent.critic.load_state_dict(agent_state[f'critic_{agent_id}'])
            agent.target_actor.load_state_dict(agent_state[f'target_actor_{agent_id}'])
            agent.target_critic.load_state_dict(agent_state[f'target_critic_{agent_id}'])
            agent._config = agent_state[f'config_{agent_id}'].get(f'config_{agent_id}', {})
            agent.__dict__.update(**agent._config)
