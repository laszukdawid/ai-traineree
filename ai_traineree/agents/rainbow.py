import copy
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from ai_traineree import DEVICE
from ai_traineree.agents import AgentBase
from ai_traineree.agents.agent_utils import soft_update
from ai_traineree.buffers.buffer_factory import BufferFactory
from ai_traineree.buffers.nstep import NStepBuffer
from ai_traineree.buffers.per import PERBuffer
from ai_traineree.loggers import DataLogger
from ai_traineree.networks.heads import RainbowNet
from ai_traineree.types import AgentState, BufferState, NetworkState
from ai_traineree.types.dataspace import DataSpace
from ai_traineree.types.experience import Experience
from ai_traineree.utils import to_numbers_seq, to_tensor


class RainbowAgent(AgentBase):
    """Rainbow agent as described in [1].

    Rainbow is a DQN agent with some improvements that were suggested before 2017.
    As mentioned by the authors it's not exhaustive improvement but all changes are in
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

    model = "Rainbow"

    def __init__(
        self,
        obs_space: DataSpace,
        action_space: DataSpace,
        state_transform: Optional[Callable] = None,
        reward_transform: Optional[Callable] = None,
        **kwargs,
    ):
        """
        A wrapper over the DQN thus majority of the logic is in the DQNAgent.
        Special treatment is required because the Rainbow agent uses categorical nets
        which operate on probability distributions. Each action is taken as the estimate
        from such distributions.

        Parameters:
            obs_space (DataSpace): Dataspace describing the input.
            action_space (DataSpace): Dataspace describing the output.
            state_transform (optional func):
            reward_transform (optional func):

        Keyword arguments:
            pre_network_fn (function that takes input_shape and returns network):
                Used to preprocess state before it is used in the value- and advantage-function in the dueling nets.
            hidden_layers (tuple of ints): Shape of the hidden layers in fully connected network. Default: (100, 100).
            lr (default: 1e-3): Learning rate value.
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
            v_min (float): Lower bound for distributional value V. Default: -10.
            v_max (float): Upper bound for distributional value V. Default: 10.
            num_atoms (int): Number of atoms (discrete states) in the value V distribution. Default: 21.

        """
        super().__init__(**kwargs)
        self.device = self._register_param(kwargs, "device", DEVICE, update=True)

        self.obs_space = obs_space
        self.action_space = action_space
        self._config["obs_space"] = self.obs_space
        self._config["action_space"] = self.action_space
        self.action_size = action_space.to_feature()

        self.lr = float(self._register_param(kwargs, "lr", 3e-4))
        self.gamma = float(self._register_param(kwargs, "gamma", 0.99))
        self.tau = float(self._register_param(kwargs, "tau", 0.002))
        self.update_freq = int(self._register_param(kwargs, "update_freq", 1))
        self.batch_size = int(self._register_param(kwargs, "batch_size", 80, update=True))
        self.buffer_size = int(self._register_param(kwargs, "buffer_size", int(1e5), update=True))
        self.warm_up = int(self._register_param(kwargs, "warm_up", 0))
        self.number_updates = int(self._register_param(kwargs, "number_updates", 1))
        self.max_grad_norm = float(self._register_param(kwargs, "max_grad_norm", 10))

        self.iteration: int = 0
        self.using_double_q = bool(self._register_param(kwargs, "using_double_q", True))

        self.state_transform = state_transform if state_transform is not None else lambda x: x
        self.reward_transform = reward_transform if reward_transform is not None else lambda x: x

        v_min = float(self._register_param(kwargs, "v_min", -10))
        v_max = float(self._register_param(kwargs, "v_max", 10))
        self.num_atoms = int(self._register_param(kwargs, "num_atoms", 21, drop=True))
        self.z_atoms = torch.linspace(v_min, v_max, self.num_atoms, device=self.device)
        self.z_delta = self.z_atoms[1] - self.z_atoms[0]

        self.buffer = PERBuffer(**kwargs)
        self.__batch_indices = torch.arange(self.batch_size, device=self.device)

        self.n_steps = int(self._register_param(kwargs, "n_steps", 3))
        self.n_buffer = NStepBuffer(n_steps=self.n_steps, gamma=self.gamma)

        # Note that in case a pre_network is provided, e.g. a shared net that extracts pixels values,
        # it should be explicitly passed in kwargs
        kwargs["hidden_layers"] = to_numbers_seq(self._register_param(kwargs, "hidden_layers", (100, 100)))
        self.net = RainbowNet(obs_space.shape, self.action_size, num_atoms=self.num_atoms, **kwargs)
        self.target_net = RainbowNet(obs_space.shape, self.action_size, num_atoms=self.num_atoms, **kwargs)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.dist_probs = None
        self._loss = float("nan")

    @property
    def loss(self):
        return {"loss": self._loss}

    @loss.setter
    def loss(self, value):
        if isinstance(value, dict):
            value = value["loss"]
        self._loss = value

    def step(self, experience: Experience) -> None:
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
        assert isinstance(experience.action, int), "Rainbow expects discrete action (int)"
        self.iteration += 1
        t_obs = to_tensor(self.state_transform(experience.obs)).float().to("cpu")
        t_next_obs = to_tensor(self.state_transform(experience.next_obs)).float().to("cpu")
        reward = self.reward_transform(experience.reward)

        # Delay adding to buffer to account for n_steps (particularly the reward)
        self.n_buffer.add(
            obs=t_obs.numpy(),
            action=[int(experience.action)],
            reward=[reward],
            done=[experience.done],
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
        """
        Returns actions for given state as per current policy.

        Parameters:
            state: Current available state from the environment.
            epislon: Epsilon value in the epislon-greedy policy.

        """
        # Epsilon-greedy action selection
        if self._rng.random() < eps:
            # TODO: Update with action_space.sample() once implemented
            assert len(self.action_space.shape) == 1, "Only 1D is supported right now"
            action = self._rng.randint(self.action_space.low, self.action_space.high)
            return experience.update(action=action)

        t_obs = to_tensor(self.state_transform(experience.obs)).float().unsqueeze(0).to(self.device)
        self.dist_probs = self.net.act(t_obs)
        q_values = (self.dist_probs * self.z_atoms).sum(-1)
        action = int(q_values.argmax(-1))  # Action maximizes state-action value Q(s, a)
        return experience.update(action=action)

    def learn(self, experiences: Dict[str, List]) -> None:
        """
        Parameters:
            experiences: Contains all experiences for the agent. Typically sampled from the memory buffer.
                Five keys are expected, i.e. `state`, `action`, `reward`, `next_state`, `done`.
                Each key contains a array and all arrays have to have the same length.

        """
        rewards = to_tensor(experiences["reward"]).float().to(self.device)
        dones = to_tensor(experiences["done"]).type(torch.int).to(self.device)
        obss = to_tensor(experiences["obs"]).float().to(self.device)
        next_obss = to_tensor(experiences["next_obs"]).float().to(self.device)
        actions = to_tensor(experiences["action"]).type(torch.long).to(self.device)
        assert rewards.shape == dones.shape == (self.batch_size, 1)
        assert obss.shape == next_obss.shape == (self.batch_size,) + self.obs_space.shape
        assert actions.shape == (self.batch_size, 1)  # Discrete domain

        with torch.no_grad():
            prob_next = self.target_net.act(next_obss)
            q_next = (prob_next * self.z_atoms).sum(-1) * self.z_delta
            if self.using_double_q:
                duel_prob_next = self.net.act(next_obss)
                a_next = torch.argmax((duel_prob_next * self.z_atoms).sum(-1), dim=-1)
            else:
                a_next = torch.argmax(q_next, dim=-1)

            prob_next = prob_next[self.__batch_indices, a_next, :]

        m = self.net.dist_projection(rewards, 1 - dones, self.gamma ** self.n_steps, prob_next)
        assert m.shape == (self.batch_size, self.num_atoms)

        log_prob = self.net(obss, log_prob=True)
        assert log_prob.shape == (self.batch_size,) + self.action_size + (self.num_atoms,)
        log_prob = log_prob[self.__batch_indices, actions.squeeze(), :]
        assert log_prob.shape == m.shape == (self.batch_size, self.num_atoms)

        # Cross-entropy loss error and the loss is batch mean
        error = -torch.sum(m * log_prob, 1)
        assert error.shape == (self.batch_size,)
        loss = error.mean()
        assert loss >= 0

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self._loss = float(loss.item())

        if hasattr(self.buffer, "priority_update"):
            assert (~torch.isnan(error)).any()
            self.buffer.priority_update(experiences["index"], error.detach().cpu().numpy())

        # Update networks - sync local & target
        soft_update(self.target_net, self.net, self.tau)

    def state_dict(self) -> Dict[str, dict]:
        """Returns agent's state dictionary.

        Returns:
            State dicrionary for internal networks.

        """
        return {"net": self.net.state_dict(), "target_net": self.target_net.state_dict()}

    def log_metrics(self, data_logger: DataLogger, step: int, full_log: bool = False):
        data_logger.log_value("loss/agent", self._loss, step)

        if full_log and self.dist_probs is not None:
            assert len(self.action_space.shape) == 1, "Only 1D actions currently supported"
            action_size = self.action_size[0]
            for action_idx in range(action_size):
                dist = self.dist_probs[0, action_idx]
                data_logger.log_value(f"dist/expected_{action_idx}", (dist * self.z_atoms).sum().item(), step)
                data_logger.add_histogram(
                    f"dist/Q_{action_idx}",
                    min=self.z_atoms[0],
                    max=self.z_atoms[-1],
                    num=len(self.z_atoms),
                    sum=dist.sum(),
                    sum_squares=dist.pow(2).sum(),
                    bucket_limits=self.z_atoms + self.z_delta,
                    bucket_counts=dist,
                    global_step=step,
                )

        # This method, `log_metrics`, isn't executed on every iteration but just in case we delay plotting weights.
        # It simply might be quite costly. Thread wisely.
        if full_log:
            for idx, layer in enumerate(self.net.value_net.layers):
                if hasattr(layer, "weight"):
                    data_logger.create_histogram(f"value_net/layer_weights_{idx}", layer.weight.cpu(), step)
                if hasattr(layer, "bias") and layer.bias is not None:
                    data_logger.create_histogram(f"value_net/layer_bias_{idx}", layer.bias.cpu(), step)
            for idx, layer in enumerate(self.net.advantage_net.layers):
                if hasattr(layer, "weight"):
                    data_logger.create_histogram(f"advantage_net/layer_{idx}", layer.weight.cpu(), step)
                if hasattr(layer, "bias") and layer.bias is not None:
                    data_logger.create_histogram(f"advantage_net/layer_bias_{idx}", layer.bias.cpu(), step)

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
        agent = RainbowAgent(**config)
        if state.network is not None:
            agent.set_network(state.network)
        if state.buffer is not None:
            agent.set_buffer(state.buffer)
        return agent

    def set_network(self, network_state: NetworkState) -> None:
        self.net.load_state_dict(network_state.net["net"])
        self.target_net.load_state_dict(network_state.net["target_net"])

    def set_buffer(self, buffer_state: BufferState) -> None:
        self.buffer = BufferFactory.from_state(buffer_state)

    def save_state(self, path: str) -> None:
        """Saves agent's state into a file.

        Parameters:
            path: String path where to write the state.

        """
        agent_state = self.get_state()
        torch.save(agent_state, path)

    def load_state(self, path: str) -> None:
        """Loads state from a file under provided path.

        Parameters:
            path: String path indicating where the state is stored.

        """
        agent_state = torch.load(path)
        self._config = agent_state.get("config", {})
        self.__dict__.update(**self._config)

        self.net.load_state_dict(agent_state["net"])
        self.target_net.load_state_dict(agent_state["target_net"])

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

    def __eq__(self, o: object) -> bool:
        return (
            super().__eq__(o)
            and isinstance(o, type(self))
            and self._config == o._config
            and self.buffer == o.buffer
            and self.get_network_state() == o.get_network_state()
        )
