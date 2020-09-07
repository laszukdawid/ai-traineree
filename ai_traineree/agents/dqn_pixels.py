from ai_traineree import DEVICE
from ai_traineree.agents.utils import soft_update
from ai_traineree.buffers import PERBuffer
from ai_traineree.networks import QNetwork2D
from ai_traineree.types import AgentType

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from typing import Dict, Optional, Sequence, Union


class DQNPixelAgent(AgentType):
    """Deep Q-Learning Network.
    Dual DQN implementation.

    Note (18/08/2020): Currently, this implemention differs very little from the `dqn.py` implemention.
    In the coming days/weeks there will be either a merge, or tuning that leads to higher separation.
    """

    name = "DQNPixels"

    def __init__(
        self, state_size: Union[Sequence[int], int], action_size: int, hidden_layers: Sequence[int]=(64, 64),
        lr: float = 0.001, gamma: float = 0.99, tau: float = 0.002, config: Optional[Dict]=None,
        device=None, **kwargs
    ):

        config = config if config is not None else {}
        self.device = device if device is not None else DEVICE
        self.state_dim = state_size if not isinstance(state_size, int) else (state_size,)
        self.state_size = 1
        self.action_size = action_size

        self.lr = float(config.get('lr', lr))
        self.gamma = float(config.get('gamma', gamma))
        self.tau = float(config.get('tau', tau))

        self.update_freq = int(config.get('update_freq', 1))
        self.batch_size = int(config.get('batch_size', 32))
        self.warm_up = int(config.get('warm_up', 0))
        self.number_updates = int(config.get('number_updates', 1))

        self.iteration: int = 0
        self.buffer = PERBuffer(self.batch_size)

        self.hidden_layers = config.get('hidden_layers', hidden_layers)
        self.qnet = QNetwork2D(self.state_dim, self.action_size, hidden_layers=self.hidden_layers).to(self.device)
        self.target_qnet = QNetwork2D(self.state_dim, self.action_size, hidden_layers=self.hidden_layers).to(self.device)
        self.optimizer = optim.AdamW(self.qnet.parameters(), lr=self.lr)

        self.writer = kwargs.get("writer")

    def step(self, state, action, reward, next_state, done) -> None:
        self.iteration += 1
        state = torch.tensor(state).unsqueeze(0).unsqueeze(0)
        next_state = torch.tensor(next_state).unsqueeze(0).unsqueeze(0)
        self.buffer.add(state=state, action=[action], reward=[reward], next_state=next_state, done=[done])

        if self.iteration < self.warm_up:
            return

        if len(self.buffer) > self.batch_size and (self.iteration % self.update_freq) == 0:
            for _ in range(self.number_updates):
                self.learn(self.buffer.sample(beta=0.4))

    def act(self, state, eps: float = 0.) -> int:
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(self.device)
        self.qnet.eval()
        with torch.no_grad():
            action_values = self.qnet(state)
        self.qnet.train()

        # Epsilon-greedy action selection
        if np.random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.randint(self.action_size)

    def learn(self, experiences) -> None:
        rewards = torch.tensor(experiences['reward']).to(self.device)
        dones = torch.tensor(experiences['done']).type(torch.int).to(self.device)
        states = torch.cat(experiences['state']).type(torch.float32).to(self.device)
        next_states = torch.cat(experiences['next_state']).type(torch.float32).to(self.device)
        actions = torch.from_numpy(np.array(experiences['action'])).to(self.device)

        Q_targets_next = self.target_qnet(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma + Q_targets_next * (1 - dones))
        Q_expected = self.qnet(states).gather(1, actions.type(torch.long).to(self.device))

        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss = loss.item()

        if hasattr(self.buffer, 'priority_update'):
            td_error = Q_expected - Q_targets
            self.buffer.priority_update(experiences['index'], 1./td_error.abs())

        # Update networks - sync local & target
        soft_update(self.target_qnet, self.qnet, self.tau)

    def describe_agent(self) -> Dict:
        """Returns agent's state dictionary."""
        return self.qnet.state_dict()

    def log_writer(self, episode):
        self.writer.add_scalar("loss/actor", self.actor_loss, episode)
        self.writer.add_scalar("loss/critic", self.critic_loss, episode)

    def save_state(self, path: str):
        agent_state = dict(net_local=self.qnet.state_dict(), net_target=self.target_qnet.state_dict())
        torch.save(agent_state, f'{path}_agent.net')

    def load_state(self, path: str):
        agent_state = torch.load(f'{path}_agent.net')
        self.qnet.load_state_dict(agent_state['qnet'])
        self.target_qnet.load_state_dict(agent_state['target_qnet'])
