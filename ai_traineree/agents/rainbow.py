from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.agents.utils import soft_update
from ai_traineree.buffers import NStepBuffer, PERBuffer, ReplayBuffer
from ai_traineree.networks import RainbowNet

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


class RainbowAgent(DQNAgent):
    """Rainbow agent as described in [1].

    **This does not work. No idea why. All seems alright(?). Help.**

    Rainbow is a DQN agent with some improvments that were suggested before 2017.
    As mentioned by the authors it's not exhaustive improvment but all changes are in
    relatively separate areas so their connection makes sense. These improvements are:
    * Priority Experience Replay
    * Multi-step
    * Double Q net
    * Dueling nets
    * NoisyNet
    * CategoricalNet for Q estimate

    Consider this class as a particular version of the DQN agent.

    [1] "Rainbow: Combining Improvements in Deep Reinforcement Learning" by Hessel et al. (DeepMind team)
    https://arxiv.org/abs/1710.02298

    """

    name = "Rainbow"

    def __init__(self, *args, **kwargs):
        """
        Accepted parameters:
        :param float lr: learning rate (default: 1e-3)
        :param float gamma: discount factor (default: 0.99)
        :param float tau: soft-copy factor (default: 0.002) 

        """
        hidden_layers = kwargs.get("hidden_layers", (300, 300))
        super(RainbowAgent, self).__init__(*args, **kwargs)

        # self.buffer = ReplayBuffer(batch_size=self.batch_size, buffer_size=self.buffer_size)
        self.buffer = PERBuffer(batch_size=self.batch_size, buffer_size=self.buffer_size)

        self.using_double_q = bool(kwargs.get("using_double_q", True))
        self.v_min = float(kwargs.get("v_min", -10))
        self.v_max = float(kwargs.get("v_max", 10))
        self.n_atoms = int(kwargs.get("n_atoms", 21))
        self.z_atoms = torch.linspace(self.v_min, self.v_max, self.n_atoms, device=self.device)
        self.z_delta = self.z_atoms[1] - self.z_atoms[0]
        
        self.batch_indices = torch.arange(self.batch_size, device=self.device)
        self.offset = torch.linspace(0, ((self.batch_size - 1) * self.n_atoms), self.batch_size, device=self.device)
        self.offset = self.offset.unsqueeze(1).expand(self.batch_size, self.n_atoms)

        self.n_steps = kwargs.get("n_steps", 3)
        self.n_buffer = NStepBuffer(n_steps=self.n_steps, gamma=self.gamma)

        self.net = RainbowNet(self.state_size[0], self.action_size, num_atoms=self.n_atoms, hidden_layers=hidden_layers, device=self.device)
        self.target_net = RainbowNet(self.state_size[0], self.action_size, num_atoms=self.n_atoms, hidden_layers=hidden_layers, device=self.device)

        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)

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
        prob = self.net.act(state)
        q = (prob * self.z_atoms).sum(-1)
        action_values = q.argmax(-1)
        return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences) -> None:
        rewards = torch.tensor(experiences['reward'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(experiences['done']).type(torch.int).to(self.device)
        states = torch.tensor(experiences['state'], dtype=torch.float32).to(self.device)
        next_states = torch.tensor(experiences['next_state'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(experiences['action'], dtype=torch.long).to(self.device)

        with torch.no_grad():
            prob_next = self.target_net.act(next_states)
            q_next = (prob_next * self.z_atoms).sum(-1) * self.z_delta
            if self.using_double_q:
                duel_prob_next = self.net.act(next_states)
                a_next = torch.argmax((duel_prob_next * self.z_atoms).sum(-1), dim=-1)
            else:
                a_next = torch.argmax(q_next, dim=-1)

            prob_next = prob_next[self.batch_indices, a_next, :]

            Tz = rewards + self.gamma ** self.n_steps * (1 - dones) * self.z_atoms.view(1, -1)
            Tz.clamp_(self.v_min, self.v_max)  # in place

            b_idx = (Tz - self.v_min) / self.z_delta
            l_idx = b_idx.floor().to(torch.int64)
            u_idx = b_idx.ceil().to(torch.int64)

            # Fix disappearing probability mass when l = b = u (b is int)
            l_idx[(u_idx > 0) * (l_idx == u_idx)] -= 1
            u_idx[(l_idx < (self.n_atoms - 1)) * (l_idx == u_idx)] += 1
            
            l_offset_idx = (l_idx + self.offset).type(torch.int64)
            u_offset_idx = (u_idx + self.offset).type(torch.int64)

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size * self.n_atoms)

            # Dealing with indices. *Note* not to forget batches.
            # m[l] = m[l] + p(s[t+n], a*)(u - b)
            m.index_add_(0, l_offset_idx.view(-1), (prob_next * (u_idx.float() - b_idx)).view(-1))
            # m[u] = m[u] + p(s[t+n], a*)(b - l)
            m.index_add_(0, u_offset_idx.view(-1), (prob_next * (b_idx - l_idx.float())).view(-1))

            m = m.view(self.batch_size, self.n_atoms)

        log_prob = self.net(states, log_prob=True)
        log_prob = log_prob[self.batch_indices, actions.squeeze(), :]

        # Cross-entropy loss error and the loss is batch mean
        error = -torch.sum(m * log_prob, 1)
        loss = error.mean()
        assert loss >= 0

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.loss = loss.item()

        if hasattr(self.buffer, 'priority_update'):
            assert (~torch.isnan(error)).any()
            self.buffer.priority_update(experiences['index'], error.detach().cpu().numpy())

        # Update networks - sync local & target
        soft_update(self.target_net, self.net, self.tau)