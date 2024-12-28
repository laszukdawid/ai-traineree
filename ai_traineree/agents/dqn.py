import copy
from typing import Callable, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ai_traineree import DEVICE
from ai_traineree.agents import AgentBase
from ai_traineree.agents.agent_utils import soft_update
from ai_traineree.buffers.buffer_factory import BufferFactory
from ai_traineree.buffers.nstep import NStepBuffer
from ai_traineree.buffers.per import PERBuffer
from ai_traineree.loggers import DataLogger
from ai_traineree.networks import NetworkType, NetworkTypeClass
from ai_traineree.networks.heads import DuelingNet
from ai_traineree.types import AgentState, BufferState, DataSpace, NetworkState
from ai_traineree.types.experience import Experience
from ai_traineree.utils import to_numbers_seq, to_tensor


class DQNAgent(AgentBase):
    """Deep Q-Learning Network (DQN).

    The agent is not a vanilla DQN, although can be configured as such.
    The default config includes dual dueling nets and the priority experience buffer.
    Learning is also delayed by slowly copying to target nets (via tau parameter).
    Although NStep is implemented the default value is 1-step reward.

    There is also a specific implementation of the DQN called the Rainbow which differs
    to this implementation by working on the discrete space projection of the Q(s,a) function.
    """

    model = "DQN"

    def __init__(
        self,
        obs_space: DataSpace,
        action_space: DataSpace,
        network_fn: Callable[[], NetworkType] = None,
        network_class: Type[NetworkTypeClass] = None,
        state_transform: Callable | None = None,
        reward_transform: Callable | None = None,
        **kwargs,
    ):
        """Initiates the DQN agent.

        Parameters:
            obs_space (DataSpace): Dataspace describing the input.
            action_space (DataSpace): Dataspace describing the output.
            network_fn (optional func): Function used to instantiate a network used by the agent.
            network_class (optional cls): Class of network that is instantiated with internal params to create network.
            state_transform (optional func): Function to transform (encode) state before used by the network.
            reward_transform (optional func): Function to transform reward before use.

        Keyword arguments:
            hidden_layers (tuple of ints): Shape of the hidden layers in fully connected network. Default: (64, 64).
            lr (float): Learning rate value. Default: 3e-4.
            gamma (float): Discount factor. Default: 0.99.
            tau (float): Soft-copy factor. Default: 0.002.
            update_freq (int): Number of steps between each learning step. Default 1.
            batch_size (int): Number of samples to use at each learning step. Default: 80.
            buffer_size (int): Number of most recent samples to keep in memory for learning. Default: 1e5.
            warm_up (int): Number of samples to observe before starting any learning step. Default: 0.
            number_updates (int): How many times to use learning step in the learning phase. Default: 1.
            max_grad_norm (float): Maximum norm of the gradient used in learning. Default: 10.
            using_double_q (bool): Whether to use Double Q Learning network. Default: True.
            n_steps (int): Number of lookahead steps when estimating reward. See :class:`NStepBuffer`. Default: 3.

        """
        super().__init__(**kwargs)

        self.device = self._register_param(kwargs, "device", DEVICE, update=True)
        self.obs_space = obs_space
        self.action_space = action_space

        self.lr = float(self._register_param(kwargs, "lr", 3e-4))  # Learning rate
        self.gamma = float(self._register_param(kwargs, "gamma", 0.99))  # Discount value
        self.tau = float(self._register_param(kwargs, "tau", 0.002))  # Soft update

        self.update_freq = int(self._register_param(kwargs, "update_freq", 1))
        self.batch_size = int(self._register_param(kwargs, "batch_size", 64, update=True))
        self.buffer_size = int(self._register_param(kwargs, "buffer_size", int(1e5), update=True))
        self.warm_up = int(self._register_param(kwargs, "warm_up", 0))
        self.number_updates = int(self._register_param(kwargs, "number_updates", 1))
        self.max_grad_norm = float(self._register_param(kwargs, "max_grad_norm", 10))

        self.iteration: int = 0
        self.buffer = PERBuffer(**kwargs)
        self.using_double_q = bool(self._register_param(kwargs, "using_double_q", True))

        self.n_steps = int(self._register_param(kwargs, "n_steps", 1))
        self.n_buffer = NStepBuffer(n_steps=self.n_steps, gamma=self.gamma)

        hidden_layers = to_numbers_seq(self._register_param(kwargs, "hidden_layers", (64, 64)))
        self.state_transform = state_transform if state_transform is not None else lambda x: x
        self.reward_transform = reward_transform if reward_transform is not None else lambda x: x
        action_feat = action_space.to_feature()

        if network_fn is not None:
            self.net = network_fn()
            self.target_net = network_fn()
        elif network_class is not None:
            self.net = network_class(obs_space.shape, action_feat, hidden_layers=hidden_layers, device=self.device)
            self.target_net = network_class(
                obs_space.shape, action_feat, hidden_layers=hidden_layers, device=self.device
            )
        else:
            self.net = DuelingNet(obs_space.shape, action_feat, hidden_layers=hidden_layers, device=self.device)
            self.target_net = DuelingNet(obs_space.shape, action_feat, hidden_layers=hidden_layers, device=self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self._loss: float = float("nan")

    @property
    def loss(self) -> dict[str, float]:
        return {"loss": self._loss}

    @loss.setter
    def loss(self, value):
        if isinstance(value, dict):
            value = value["loss"]
        self._loss = value

    def __eq__(self, o: object) -> bool:
        return (
            super().__eq__(o)
            and isinstance(o, type(self))
            and self._config == o._config
            and self.buffer == o.buffer
            and self.n_buffer == o.n_buffer
            and self.get_network_state() == o.get_network_state()
        )

    def reset(self):
        self.net.reset_parameters()
        self.target_net.reset_parameters()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def step(self, exp: Experience) -> None:
        """Letting the agent to take a step.

        On some steps the agent will initiate learning step. This is dependent on
        the `update_freq` value.

        Parameters:
            obs (ObservationType): Observation.
            action (int): Discrete action associated with observation.
            reward (float): Reward obtained for taking action at state.
            next_obs (ObservationType): Observation in a state where the action took.
            done: (bool) Whether in terminal (end of episode) state.

        """
        assert isinstance(exp.action, int), "DQN expects discrete actions (int)"
        self.iteration += 1
        t_obs = to_tensor(self.state_transform(exp.obs)).float().to("cpu")
        t_next_obs = to_tensor(self.state_transform(exp.next_obs)).float().to("cpu")
        reward = self.reward_transform(exp.reward)

        # Delay adding to buffer to account for n_steps (particularly the reward)
        self.n_buffer.add(
            obs=t_obs.numpy(),
            action=[int(exp.action)],
            reward=[reward],
            done=[exp.done],
            next_obs=t_next_obs.numpy(),
        )
        if not self.n_buffer.available:
            return

        self.buffer.add(**self.n_buffer.get().get_dict())

        if self.iteration < self.warm_up:
            return

        if len(self.buffer) >= self.batch_size and (self.iteration % self.update_freq) == 0:
            for _ in range(self.number_updates):
                self.learn(self.buffer.sample())

            # Update networks only once - sync local & target
            soft_update(self.target_net, self.net, self.tau)

    def act(self, experience: Experience, eps: float = 0.0) -> Experience:
        """Returns actions for given obs as per current policy.

        Parameters:
            experience (Experience): current observation
            eps (optional float): epsilon, for epsilon-greedy action selection. Default 0.

        Returns:
            Categorical value for the action.

        """
        # Epsilon-greedy action selection
        if self._rng.random() < eps:
            action = self._rng.randint(self.action_space.low, self.action_space.high)
            return experience.update(action=action)

        t_obs = to_tensor(self.state_transform(experience.obs)).float()
        t_obs = t_obs.unsqueeze(0).to(self.device)
        action_values = self.net.act(t_obs)
        action = int(torch.argmax(action_values.cpu()))
        return experience.update(action=action)

    def learn(self, experiences: dict[str, list]) -> None:
        """Updates agent's networks based on provided experience.

        Parameters:
            experiences: Samples experiences from the experience buffer.

        """
        rewards = to_tensor(experiences["reward"]).type(torch.float32).to(self.device)
        dones = to_tensor(experiences["done"]).type(torch.int).to(self.device)
        obss = to_tensor(experiences["obs"]).type(torch.float32).to(self.device)
        next_obss = to_tensor(experiences["next_obs"]).type(torch.float32).to(self.device)
        actions = to_tensor(experiences["action"]).type(torch.long).to(self.device)

        with torch.no_grad():
            Q_targets_next = self.target_net.act(next_obss).detach()
            if self.using_double_q:
                _a = torch.argmax(self.net(next_obss), dim=-1).unsqueeze(-1)
                max_Q_targets_next = Q_targets_next.gather(1, _a)
            else:
                max_Q_targets_next = Q_targets_next.max(1)[0].unsqueeze(1)
        Q_targets = rewards + self.n_buffer.n_gammas[-1] * max_Q_targets_next * (1 - dones)
        Q_expected: torch.Tensor = self.net(obss).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self._loss = float(loss.item())

        if hasattr(self.buffer, "priority_update"):
            error = Q_expected - Q_targets
            assert any(~torch.isnan(error))
            self.buffer.priority_update(experiences["index"], error.abs())

    def state_dict(self) -> dict[str, dict]:
        """Describes agent's networks.

        Returns:
            state: (dict) Provides actors and critics states.

        """
        return {
            "net": self.net.state_dict(),
            "target_net": self.target_net.state_dict(),
        }

    def log_metrics(self, data_logger: DataLogger, step: int, full_log: bool = False):
        """Uses provided DataLogger to provide agent's metrics.

        Parameters:
            data_logger (DataLogger): Instance of the SummaryView, e.g. torch.utils.tensorboard.SummaryWritter.
            step (int): Ordering value, e.g. episode number.
            full_log (bool): Whether to all available information. Useful to log with lesser frequency.
        """
        data_logger.log_value("loss/agent", self._loss, step)

    def get_state(self) -> AgentState:
        """Provides agent's internal state."""
        return AgentState(
            model=self.model,
            obs_space=self.obs_space,
            action_space=self.action_space,
            config=self._config,
            buffer=copy.deepcopy(self.buffer.get_state()),
            network=copy.deepcopy(self.get_network_state()),
        )

    def get_network_state(self) -> NetworkState:
        return NetworkState(net=dict(net=self.net.state_dict(), target_net=self.target_net.state_dict()))

    @staticmethod
    def from_state(state: AgentState) -> AgentBase:
        config = copy.copy(state.config)
        config.update({"obs_space": state.obs_space, "action_space": state.action_space})
        agent = DQNAgent(**config)
        if state.network is not None:
            agent.set_network(state.network)
        if state.buffer is not None:
            agent.set_buffer(state.buffer)
        return agent

    def set_buffer(self, buffer_state: BufferState) -> None:
        self.buffer = BufferFactory.from_state(buffer_state)

    def set_network(self, network_state: NetworkState) -> None:
        self.net.load_state_dict(network_state.net["net"])
        self.target_net.load_state_dict(network_state.net["target_net"])

    def save_state(self, path: str):
        """Saves agent's state into a file.

        Parameters:
            path: String path where to write the state.

        """
        agent_state = self.get_state()
        torch.save(agent_state, path)

    def load_state(self, *, path: str | None = None, state: AgentState | None = None) -> None:
        """Loads state from a file under provided path.

        Parameters:
            path: String path indicating where the state is stored.

        """
        if path is None and state is None:
            raise ValueError("Either `path` or `state` must be provided to load agent's state.")
        if path is not None:
            state = torch.load(path)

        # Populate agent
        agent_state = state.agent
        self._config = agent_state.config
        self.__dict__.update(**self._config)

        # Populate network
        network_state = state.network
        self.net.load_state_dict(network_state.net["net"])
        self.target_net.load_state_dict(network_state.net["target_net"])
        self.buffer = PERBuffer(**self._config)

    def save_buffer(self, path: str) -> None:
        """Saves data from the buffer into a file under provided path.

        Parameters:
            path: String path where to write the buffer.

        """
        import json

        dump = self.buffer.dump_buffer(serialize=True)
        with open(path, "w") as f:
            json.dump(dump, f)

    def load_buffer(self, path: str) -> None:
        """Loads data into the buffer from provided file path.

        Parameters:
            path: String path indicating where the buffer is stored.

        """
        import json

        with open(path, "r") as f:
            buffer_dump = json.load(f)
        self.buffer.load_buffer(buffer_dump)
