from ai_traineree.types import AgentType, TaskType
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from ai_traineree.buffers import ReplayBuffer
from ai_traineree.networks import QNetwork


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(AgentType):

    name = "DQN"

    def __init__(
            self, state_size: int, action_size: int, batch_size: int = 32, update_freq: int = 4, hidden_layers=(64, 64),
            lr: float = 0.01, gamma: float = 0.99, tau: float = 0.002, device=None):
        self.state_size = state_size
        self.action_size = action_size

        self.lr = lr
        self.gamma = gamma
        self.tau = tau

        self.update_freq = update_freq
        self.batch_size = batch_size

        self.device = device if device is not None else DEVICE

        self.t_step = 0
        self.memory = ReplayBuffer(self.batch_size)
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, hidden_layers=hidden_layers).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, hidden_layers=hidden_layers).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        self.last_loss = None

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_freq time steps.
        self.t_step = (self.t_step + 1) % self.update_freq
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                self.learn(self.memory.sample())

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
        self.soft_update()

    def soft_update(self):
        zipped_params = zip(self.qnetwork_local.parameters(), self.qnetwork_target.parameters())
        for local_param, target_param in zipped_params:
            # target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*local_param.data)
            target_param.data = self.tau * local_param.data + (1.0 - self.tau) * local_param.data

    def describe_agent(self):
        return self.qnetwork_local.state_dict()
