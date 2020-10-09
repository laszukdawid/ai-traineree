from ai_traineree import DEVICE
from ai_traineree.agents.utils import soft_update
from ai_traineree.buffers import PERBuffer
from ai_traineree.networks import QNetwork, NetworkType
from ai_traineree.types import AgentType

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from typing import Callable, Dict, Optional, Sequence, Union


class DQNAgent(AgentType):
    """Deep Q-Learning Network.
    Dual DQN implementation.
    """

    name = "DQN"

    def __init__(
        self, state_size: Union[Sequence[int], int], action_size: int,
        lr: float = 0.001, gamma: float = 0.99, tau: float = 0.002,
        network_fn: Callable[[], NetworkType]=None,
        hidden_layers: Sequence[int]=(64, 64),
        state_transform: Optional[Callable]=None,
        reward_transform: Optional[Callable]=None,
        device=None, **kwargs
    ):
        """
        Accepted parameters:
        :param float lr: learning rate (default: 1e-3)
        :param float gamma: discount factor (default: 0.99)
        :param float tau: soft-copy factor (default: 0.002) 

        """

        self.device = device if device is not None else DEVICE
        self.state_dim = state_size if not isinstance(state_size, int) else (state_size,)
        self.state_size = 1
        self.action_size = action_size

        self.lr = float(kwargs.get('lr', lr))
        self.gamma = float(kwargs.get('gamma', gamma))
        self.tau = float(kwargs.get('tau', tau))

        self.update_freq = int(kwargs.get('update_freq', 1))
        self.batch_size = int(kwargs.get('batch_size', 32))
        self.warm_up = int(kwargs.get('warm_up', 0))
        self.number_updates = int(kwargs.get('number_updates', 1))

        self.iteration: int = 0
        self.buffer = PERBuffer(self.batch_size)

        self.state_transform = state_transform if state_transform is not None else lambda x: x
        self.reward_transform = reward_transform if reward_transform is not None else lambda x: x
        if network_fn:
            self.net = network_fn().to(self.device)
            self.target_net = network_fn().to(self.device)
        else:
            hidden_layers = kwargs.get('hidden_layers', hidden_layers)
            self.net = QNetwork(self.state_dim, self.action_size, hidden_layers=hidden_layers).to(self.device)
            self.target_net = QNetwork(self.state_dim, self.action_size, hidden_layers=hidden_layers).to(self.device)
        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.lr)

    def step(self, state, action, reward, next_state, done) -> None:
        self.iteration += 1
        state = self.state_transform(state)
        next_state = self.state_transform(state)
        reward = self.reward_transform(reward)
        self.buffer.add(state=state, action=[action], reward=[reward], next_state=next_state, done=[done])

        if self.iteration < self.warm_up:
            return

        if len(self.buffer) > self.batch_size and (self.iteration % self.update_freq) == 0:
            for _ in range(self.number_updates):
                self.learn(self.buffer.sample(beta=0.5))

    def act(self, state, eps: float = 0.) -> int:
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Epsilon-greedy action selection
        if np.random.random() < eps:
            return np.random.randint(self.action_size)

        state = self.state_transform(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_values = self.net.act(state)
        return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences) -> None:
        rewards = torch.tensor(experiences['reward']).to(self.device)
        dones = torch.tensor(experiences['done']).type(torch.int).to(self.device)
        states = torch.tensor(experiences['state'], dtype=torch.float32).to(self.device)
        next_states = torch.tensor(experiences['next_state'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(experiences['action'], dtype=torch.long).to(self.device)

        Q_targets_next = self.target_net.act(next_states).max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma + Q_targets_next * (1 - dones))
        Q_expected = self.net(states).gather(1, actions)

        loss = F.smooth_l1_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss = loss.item()

        if hasattr(self.buffer, 'priority_update'):
            td_error = Q_expected - Q_targets
            self.buffer.priority_update(experiences['index'], 1./td_error.abs())

        # Update networks - sync local & target
        soft_update(self.target_net, self.net, self.tau)

    def describe_agent(self) -> Dict:
        """Returns agent's state dictionary."""
        return self.net.state_dict()

    def log_writer(self, writer, episode):
        writer.add_scalar("loss/actor", self.actor_loss, episode)
        writer.add_scalar("loss/critic", self.critic_loss, episode)

    def save_state(self, path: str):
        agent_state = dict(net=self.net.state_dict(), target_net=self.target_net.state_dict())
        torch.save(agent_state, path)

    def load_state(self, path: str):
        agent_state = torch.load(path)
        self.net.load_state_dict(agent_state['net'])
        self.target_net.load_state_dict(agent_state['target_net'])
