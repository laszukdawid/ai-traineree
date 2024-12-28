import logging
from collections import deque
from functools import cached_property, reduce
from operator import mul
from typing import Any, Callable, Sequence

import numpy as np
import torch

from ai_traineree.types import ActionType, DataSpace, MultiAgentTaskType, StateType, TaskType

try:
    import gymnasium as gym

    BaseEnv = gym.Env  # To satisfy parser on MultiAgentUnityTask import
except ImportError:
    logging.warning("Coulnd't import `gym`. Please install `pip install -e .[gym]` if you intend to use it.")

try:
    from gym import spaces
    from gym_unity.envs import ActionFlattener
    from mlagents_envs.base_env import BaseEnv, DecisionSteps, TerminalSteps
except (ImportError, ModuleNotFoundError):
    logging.warning("Couldn't import `gym_unity` and/or `mlagents`. MultiAgentUnityTask won't work.")


GymStepResult = tuple[np.ndarray, float, bool, dict]


class TerminationMode:
    ALL = "all"
    ANY = "any"
    MAJORITY = "majority"


class GymTask(TaskType):
    logger = logging.getLogger("GymTask")

    def __init__(
        self,
        env: str | gym.Env,
        state_transform: Callable | None = None,
        reward_transform: Callable | None = None,
        can_render=True,
        stack_frames: int = 1,
        skip_start_frames: int = 0,
        **kwargs,
    ):
        """
        Parameters:
            env (gym-like env instance or str):
                Something one might get via `env = gym.make('CartPole-v0')` where `gym` is OpenAI gym compatible.
                If `env` is passed as a string then it is assumed to be a registred Gym with OpenAI interface.
                In such a case, we got you.
            state_transform (function): Default: `None`.
                Function that transform state before it's returned to the observer(s).
            reward_transform (function): Default: `None`.
                Function that shapes reward before it's returned to the observer(s).
                All arguments are expected to be named; supported names: state, action, reward, done, info.
            can_render (bool): Default: `True`.
                Whether the task can return task state (different than the step observation).
                Most common case is to provide the game view as the user would have.
                By default this flag is set to True since the most common use case is OpenAI gym, specifically
                Atari games.
            stack_frames (int): Default: 1.
                Number of frames to return when performing a step.
                By default it only returns current observation (MDP).
                When greater than 1, the returned observation will incude previous observations.
            skip_start_frames (int): Default: 0.
                Often referred as "noop frames". Indicates how many initial frames to skip.
                Every `reset()` will skip a random number of frames in range`[0, skip_start_frames]`.

        Example:
            >>> def reward_transform(*, reward, state, done):
            ...     return reward + 100*done - state[0]*0.1
            >>> task = GymTask(env='CartPole-v1', reward_transform=reward_transform)

        """
        if isinstance(env, str):
            self.name = env
            self.env = gym.make(env)
        else:
            self.name = "custom"
            self.env = env
        self.can_render = can_render
        self.is_discrete = "Discrete" in str(type(self.env.action_space))

        obs_space = self.env.observation_space.shape
        self.obs_size: int = reduce(mul, obs_space)
        self.action_size = self.__determine_action_size(self.env.action_space)
        self.state_transform = state_transform
        self.reward_transform = reward_transform

        self.stacked_frames = deque(maxlen=stack_frames)
        self.skip_start_frames = skip_start_frames

        self.seed(kwargs.get("seed"))

    @staticmethod
    def __determine_action_size(action_space):
        if "Discrete" in str(type(action_space)):
            return action_space.n
        else:
            return sum(action_space.shape)

    @property
    def obs_space(self) -> DataSpace:
        return DataSpace.from_gym_space(self.env.observation_space)

    @property
    def action_space(self) -> DataSpace:
        return DataSpace.from_gym_space(self.env.action_space)

    @property
    def actual_obs_size(self) -> int:
        return reduce(mul, self.reset().shape)

    @property
    def actual_obs_shape(self) -> Sequence[int]:
        return self.reset().shape

    def seed(self, seed):
        if isinstance(seed, (int, float)):
            return self.env.reset(seed=seed)

    def reset(self) -> torch.Tensor | np.ndarray:
        # TODO: info is currently ignored
        state, info = self.env.reset()
        # state = self.env.reset()
        if self.state_transform is not None:
            state = self.state_transform(state)

        return state

    def render(self, mode="rgb_array"):
        if self.can_render:
            # In case of OpenAI, mode can be ['human', 'rgb_array']
            return self.env.render(mode=mode)
        else:
            self.logger.warning("Asked for rendering but it's not available in this environment")
            return

    def step(self, action: ActionType) -> tuple:
        """Each action results in a new state, reward, done flag, and info about env.

        Parameters:
            action: An action that the agent is taking in current environment step.

        Returns:
            step_tuple (tuple[torch.Tensor, float, bool, Any]):
                The return consists of a next state, a reward in that state,
                a flag whether the next state is terminal and additional information provided
                by the environment regarding that state.

        """
        if self.is_discrete:
            action = int(action)
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        # Gym deprecated
        # state, reward, done, info = self.env.step(action)
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if self.state_transform is not None:
            state = self.state_transform(state)
        if self.reward_transform is not None:
            reward = self.reward_transform(reward=reward)

        if self.stacked_frames.maxlen > 1:
            self.stacked_frames.append(state)
            state = torch.tensor(self.stacked_frames).squeeze(1)  # TODO: Maybe there's some stacking?
        return (state, reward, done, info)


class PettingZooTask(MultiAgentTaskType):
    def __init__(self, env) -> None:
        """Wrapper around PettingZoo's envs to make it more compatible with EnvRunners.

        *Note*: Direct access to wrapped env is through `self.env`.

        Parameters:
            env: An instance of PettingZoo env.

        Example:
            >>> from pettingzoo.butterfly import prison_v2 as prison
            >>> env = prison.env()
            >>> task = PettingZooTask(env)
            >>> assert env == task.env
        """
        super().__init__()
        self.env = env
        self.name = "CUSTOM"
        self.agent_iter = self.env.agent_iter

    @cached_property
    def agents(self):
        return self.env.agents

    @cached_property
    def observation_spaces(self) -> dict[str, DataSpace]:
        spaces = {}
        for unit, space in self.env.observation_spaces.items():
            if type(space).__name__ == "Dict":
                space = space["observation"]
            spaces[unit] = DataSpace.from_gym_space(space)
        return spaces

    @cached_property
    def action_spaces(self) -> dict[str, DataSpace]:
        return {unit: DataSpace.from_gym_space(space) for (unit, space) in self.env.action_spaces.items()}

    def action_mask_spaces(self) -> dict[str, DataSpace] | None:
        spaces = {}
        for unit, space in self.env.observation_spaces.items():
            if not type(space).__name__ == "Dict":
                return None
            spaces[unit] = DataSpace.from_gym_space(space["action_mask"])
        return spaces

    @property
    def obs_size(self):
        return self.env.observe(self.env.agents[0]).shape

    @property
    def action_size(self):
        return self.env.action_spaces[self.env.agents[0]]

    @property
    def num_agents(self):
        return self.env.num_agents

    @property
    def is_all_done(self):
        return all(self.env.dones)

    @property
    def dones(self):
        return self.env.dones

    def last(self, agent_name: str | None = None) -> tuple[Any, float, bool, Any]:
        if agent_name is None:
            return self.env.last()
        return (
            self.env.observe(agent_name),
            self.env.rewards[agent_name],
            self.env.dones[agent_name],
            self.env.infos[agent_name],
        )

    def reset(self):
        self.env.reset()

    def step(self, action):
        self.env.step(np.array(action))
        return self.env.last()

    def render(self, mode: str):
        self.env.render(mode)

    def seed(self, seed: int) -> None:
        self.env.seed(seed)


class MultiAgentUnityTask(MultiAgentTaskType):
    """Based on `UnityToGymWrapper` from the Unity's ML-Toolkits (permalink_).


    At the time of writting the official package doesn't support multi agents.
    Until it's clear why it doesn't support (https://github.com/Unity-Technologies/ml-agents/issues/4120)
    and whether they plan on adding anything, we're keeping this version. When the fog of unknown
    has been blown away, we might consider doing a Pull Request to `ml-agents`.

    .. _permalink: https://github.com/Unity-Technologies/ml-agents/blob/3e48be4e1304d8cbbb43d8ffc335f8037cfe6f1d/gym-unity/gym_unity/envs/__init__.py#L27

    """  # noqa

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        unity_env: BaseEnv,
        uint8_visual: bool = False,
        flatten_branched: bool = False,
        allow_multiple_obs: bool = False,
        termination_mode: str = TerminationMode.ANY,
    ):
        """

        Parameters:
            unity_env: The Unity BaseEnv to be wrapped in the gym. Will be closed when the UnityToGymWrapper closes.
            uint8_visual : Return visual observations as uint8 (0-255) matrices instead of float (0.0-1.0).
                If True, turn branched discrete action spaces into a Discrete space rather than MultiDiscrete.
            allow_multiple_obs: If True, return a list of np.ndarrays as observations with the first elements
                containing the visual observations and the last element containing the array of vector observations.
                If False, returns a single np.ndarray containing either only a single visual observation or the array of
                vector observations.
            termination_mode: A string (enum) suggesting when to end an episode. Supports "any", "majority" and "all"
                which are atributes on `TerminationMode`.
        """
        self._env = unity_env

        # Take a single step so that the brain information will be sent over
        if not self._env.behavior_specs:
            self._env.step()

        self.visual_obs = None

        # Save the step result from the last time all Agents requested decisions.
        self._previous_decision_step: DecisionSteps = None
        self._flattener = None
        # Hidden flag used by Atari environments to determine if the game is over
        self.game_over = False
        self._allow_multiple_obs = allow_multiple_obs

        # When to stop the game. Only `any` supported currently but it should have options for `all` and `majority`.
        assert termination_mode in TerminationMode.__dict__
        self.termination_mode = termination_mode

        agent_name = list(self._env.behavior_specs.keys())[0]
        self.name = list(self._env.behavior_specs.keys())[0]  # TODO: no need for self.name
        self.agent_prefix = agent_name[: agent_name.index("=") + 1]
        self.group_spec = self._env.behavior_specs[agent_name]

        if self._get_n_vis_obs() == 0 and self._get_vec_obs_size() == 0:
            raise ValueError("There are no observations provided by the environment.")

        if not self._get_n_vis_obs() >= 1 and uint8_visual:
            self.logger.warning(
                "uint8_visual was set to true, but visual observations are not in use. "
                "This setting will not have any effect."
            )
        else:
            self.uint8_visual = uint8_visual
        if self._get_n_vis_obs() + self._get_vec_obs_size() >= 2 and not self._allow_multiple_obs:
            self.logger.warning(
                "The environment contains multiple observations. "
                "You must define allow_multiple_obs=True to receive them all. "
                "Otherwise, only the first visual observation (or vector observation if"
                "there are no visual observations) will be provided in the observation."
            )

        # Check for number of agents in scene.
        self._env.reset()
        decision_steps, _ = self._env.get_steps(agent_name)
        # self.num_agents = len(decision_steps)  # NOTE: Worked with FoodCollector
        self.num_agents = len(self._env.behavior_specs)
        self._previous_decision_step = decision_steps

        # Set action spaces
        if self.group_spec.is_action_discrete():
            branches = self.group_spec.discrete_action_branches
            if self.group_spec.action_size == 1:
                self._action_space = spaces.Discrete(branches[0])
            else:
                if flatten_branched:
                    self._flattener = ActionFlattener(branches)
                    self._action_space = self._flattener.action_space
                else:
                    self._action_space = spaces.MultiDiscrete(branches)

        else:
            if flatten_branched:
                self.logger.warning("The environment has a non-discrete action space. It will " "not be flattened.")
            high = np.ones(self.group_spec.action_shape)
            self._action_space = spaces.Box(-high, high, dtype=np.float32)

        # Set observations space
        list_spaces: list[gym.Space] = []
        shapes = self._get_vis_obs_shape()
        for shape in shapes:
            if uint8_visual:
                list_spaces.append(spaces.Box(0, 255, dtype=np.uint8, shape=shape))
            else:
                list_spaces.append(spaces.Box(0, 1, dtype=np.float32, shape=shape))
        if self._get_vec_obs_size() > 0:
            # vector observation is last
            high = np.array([np.inf] * self._get_vec_obs_size())
            list_spaces.append(spaces.Box(-high, high, dtype=np.float32))
        if self._allow_multiple_obs:
            self._observation_space = spaces.Tuple(list_spaces)
        else:
            self._observation_space = list_spaces[0]  # only return the first one

    # def reset(self) -> Union[List[np.ndarray], np.ndarray]:
    def reset(self) -> list[StateType]:
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object/list): the initial observation of the
        space.
        """
        self._env.reset()
        states = []
        for agent_id in range(self.num_agents):
            decision_step, _ = self._env.get_steps(self.agent_prefix + str(agent_id))
            self.game_over = False

            res: GymStepResult = self._single_step(decision_step)
            states.append(res[0])  # res contains tuple with `state` on first pos
        return states
        # return res[0]

    def step(self, action: list[Any], agent_id: int) -> GymStepResult:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

        Parameters:
            action (object/list): an action provided by the environment

        Returns:
            observation (object/list): agent's observation of the current environment
            reward (float/list) : amount of reward returned after previous action
            done (boolean/list): whether the episode has ended.
            info (dict): contains auxiliary diagnostic information.
        """
        if self._flattener is not None:
            # Translate action into list
            action = self._flattener.lookup_action(action)

        agent_name = self.agent_prefix + str(agent_id)
        self._env.set_actions(agent_name, action)
        self._env.step()
        decision_step, terminal_step = self._env.get_steps(agent_name)
        if self.detect_game_over(terminal_step):
            self.game_over = True
            out = self._single_step(terminal_step)
            self.reset()  # TODO: This is a hack to allow remaining agents to "do something". Remove!
            return out
        else:
            return self._single_step(decision_step)

    # def detect_game_over(self, termianl_steps: List[TerminalSteps]) -> bool:
    def detect_game_over(self, termianl_steps: list) -> bool:
        """Determine whether the episode has finished.

        Expects the `terminal_steps` to contain only steps that terminated. Note that other steps
        are possible in the same iteration.
        This is to keep consistent with Unity's framework but likely will go through refactoring.
        """
        if self.termination_mode == TerminationMode.ANY and len(termianl_steps) > 0:
            return True
        elif self.termination_mode == TerminationMode.MAJORITY and len(termianl_steps) > 0.5 * self.num_agents:
            return True
        elif self.termination_mode == TerminationMode.ALL and len(termianl_steps) == self.num_agents:
            return True
        else:
            return False

    # def _single_step(self, info: Union[DecisionSteps, TerminalSteps]) -> GymStepResult:
    def _single_step(self, info) -> GymStepResult:
        if self._allow_multiple_obs:
            visual_obs = self._get_vis_obs_list(info)
            visual_obs_list = []
            for obs in visual_obs:
                visual_obs_list.append(self._preprocess_single(obs[0]))
            default_observation = visual_obs_list
            if self._get_vec_obs_size() >= 1:
                default_observation.append(self._get_vector_obs(info)[0, :])
        else:
            if self._get_n_vis_obs() >= 1:
                visual_obs = self._get_vis_obs_list(info)
                default_observation = self._preprocess_single(visual_obs[0][0])
            else:
                default_observation = self._get_vector_obs(info)[0, :]

        if self._get_n_vis_obs() >= 1:
            visual_obs = self._get_vis_obs_list(info)
            self.visual_obs = self._preprocess_single(visual_obs[0][0])

        done = isinstance(info, TerminalSteps)

        return (default_observation, info.reward[0], done, {"step": info})

    def _preprocess_single(self, single_visual_obs: np.ndarray) -> np.ndarray:
        if self.uint8_visual:
            return (255.0 * single_visual_obs).astype(np.uint8)
        else:
            return single_visual_obs

    def _get_n_vis_obs(self) -> int:
        result = 0
        for shape in self.group_spec.observation_shapes:
            if len(shape) == 3:
                result += 1
        return result

    def _get_vis_obs_shape(self) -> list[tuple]:
        result: list[tuple] = []
        for shape in self.group_spec.observation_shapes:
            if len(shape) == 3:
                result.append(shape)
        return result

    @staticmethod
    # def _get_vis_obs_list(step_result: Union[DecisionSteps, TerminalSteps]) -> List[np.ndarray]:
    def _get_vis_obs_list(step_result) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 4:
                result.append(obs)
        return result

    @staticmethod
    # def _get_vector_obs(step_result: Union[DecisionSteps, TerminalSteps]) -> np.ndarray:
    def _get_vector_obs(step_result) -> np.ndarray:
        result: list[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 2:
                result.append(obs)
        return np.concatenate(result, axis=1)

    def _get_vec_obs_size(self) -> int:
        result = 0
        for shape in self.group_spec.observation_shapes:
            if len(shape) == 1:
                result += shape[0]
        return result

    def render(self, mode="rgb_array"):
        """Depending on the mode it will render the scene and either return it, or display.

        Parameters:
            mode: Currently only `rgb_array` (default) is supported.

        Returns:
            A tensor containing rendered scene. If asked mode is not supported, None is returned.

        """
        if mode != "rgb_array":
            self.logger.warning("Mode provide is '%s' but only `rgb_array` mode is supported.", mode)
            return
        return self.visual_obs

    def close(self) -> None:
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self._env.close()

    def seed(self, seed: Any = None) -> None:
        """Sets the seed for this env's random number generator(s).
        Currently not implemented.
        """
        self.logger.warning("Could not seed environment %s", self.name)
        return

    @property
    def metadata(self):
        return {"render.modes": ["rgb_array"]}

    @property
    def reward_range(self) -> tuple[float, float]:
        return -float("inf"), float("inf")

    @property
    def spec(self):
        return None

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space
