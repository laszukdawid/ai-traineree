import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from ai_traineree.buffers import ReplayBuffer
from ai_traineree.networks import QNetwork


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, env, device=None):
        self.state_size = state_size = sum(env.observation_space.shape)
        self.action_size = action_size = env.action_space.n

        self.lr = 0.005
        self.gamma = 0.98
        self.tau = 0.001

        self.update_freq = 8
        self.batch_size = 32

        self.device = device if device is not None else DEVICE

        self.t_step = 0
        # self.memory = ReplyBuffer(self.batch_size)
        self.memory = ReplayBuffer(self.batch_size)
        self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)


    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_freq time steps.
        self.t_step = (self.t_step + 1) % self.update_freq
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                self.learn(self.memory.sample())

    def act(self, state, eps: float=0.):
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

        ## Update networks - sync local & target
        self.soft_update()

    def soft_update(self):
        zipped_params = zip(self.qnetwork_local.parameters(), self.qnetwork_target.parameters())
        for local_param, target_param in zipped_params:
            # target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*local_param.data)
            target_param.data = self.tau*local_param.data + (1.0-self.tau)*local_param.data
