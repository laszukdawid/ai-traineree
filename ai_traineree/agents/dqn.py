from ai_traineree.agents.utils import soft_update
from ai_traineree.buffers import ReplayBuffer
from ai_traineree.networks import QNetwork
from ai_traineree.types import AgentType

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from typing import Dict, Optional


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent(AgentType):

    name = "DQN"

    def __init__(
            self, state_size: int, action_size: int, hidden_layers=(64, 64),
            lr: float = 0.001, gamma: float = 0.99, tau: float = 0.002, config: Optional[Dict]={}, device=None):
        self.state_size = state_size
        self.action_size = action_size

        self.lr = lr
        self.gamma = gamma
        self.tau = tau

        self.update_freq = config.get('update_freq', 1)
        self.batch_size = config.get('batch_size', 32)
        self.warm_up = config.get('warm_up', 0)
        self.number_updates = config.get('number_updates', 1)

        self.device = device if device is not None else DEVICE

        self.iteration = 0
        self.buffer = ReplayBuffer(self.batch_size)
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, hidden_layers=hidden_layers).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, hidden_layers=hidden_layers).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        self.last_loss = np.inf

    def step(self, state, action, reward, next_state, done):
        self.iteration += 1
        self.buffer.add(state, action, reward, next_state, done)

        if self.iteration < self.warm_up:
            return

        if len(self.buffer) > self.batch_size and (self.iteration % self.update_freq) == 0:
            for _ in range(self.number_updates):
                self.learn(self.buffer.sample())

    def act(self, state, eps: float = 0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if np.random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.randint(self.action_size)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma + Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.last_loss = loss.item()

        # Update networks - sync local & target
        soft_update(self.qnetwork_target, self.qnetwork_local, self.tau)

    def describe_agent(self):
        return self.qnetwork_local.state_dict()

    def save_state(self, path: str):
        agent_state = dict(net_local=self.qnetwork_local.state_dict(), net_target=self.qnetwork_target.state_dict())
        torch.save(agent_state, f'{path}_agent.net')

    def load_state(self, path: str):
        agent_state = torch.load(f'{path}_agent.net')
        self.qnetwork_local.load_state_dict(agent_state['net_local'])
        self.qnetwork_target.load_state_dict(agent_state['net_target'])
