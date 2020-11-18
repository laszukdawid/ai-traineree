import gym
import logging
import torch
from ai_traineree.types import ActionType, MultiAgentTaskType, StateType, TaskType

from typing import Callable, Optional, Tuple

# TODO: Make this optional
import numpy as np
from typing import Any, Dict, List, Union
from gym import spaces
from gym_unity.envs import UnityGymException, ActionFlattener
from mlagents_envs.base_env import BaseEnv, DecisionSteps, TerminalSteps


GymStepResult = Tuple[np.ndarray, float, bool, Dict]


class TerminationMode:
    ALL = 'all'
    ANY = 'any'
    MAJORITY = 'majority'


class GymTask(TaskType):
    def __init__(self, *, env=None, env_name: Optional[str]=None, state_transform: Optional[Callable]=None, reward_transform: Optional[Callable]=None, can_render=True):
        """
        Parameters
        ----------
        env : gym-like env object (default: None)
            Something one might get via `env = gym.make('CartPole-v0')` where `gym` is OpenAI gym compatible.
            *Note* that either `env` or `env_name` needs to be provided.
        env_name : str (default: None)
            If no `env` is provided then `env_name` is used to import registred gym.
            *Note* that either `env` or `env_name` needs to be provided.
        state_transform : function (default: None)
            Function that transform state before it's returned to the observer(s).
        reward_transform : function (default: None)
            Function that shapes reward before it's returned to the observer(s).
            All arguments are expected to be named; supported names: state, action, reward, done, info.
            >>> def reward_transform(*, reward, state, done):
            ...     return reward + 100*done - state[0]*0.1
            >>> 
        :param reward_transform: Callable[[*, state, reward, action]] -> float
            A reward shaping function. All of its arguments have to be named.
            >>> reward_transform(state=state, reward=reward, action=action)
        """
        if env is not None:
            self.name = "custom"
            self.env = env
        else:
            self.name = env_name
            self.env = gym.make(env_name)
        self.can_render = can_render
        self.is_discrete = "Discrete" in str(type(self.env.action_space))

        state_shape = self.env.observation_space.shape
        self.state_size = state_shape[0] if len(state_shape) == 1 else state_shape
        self.action_size = self.__determine_action_size(self.env.action_space)
        self.state_transform = state_transform
        self.reward_transform = reward_transform

    @staticmethod
    def __determine_action_size(action_space):
        if "Discrete" in str(type(action_space)):
            return action_space.n
        else:
            return sum(action_space.shape)

    @property
    def actual_state_size(self):
        state = self.reset()
        return state.shape

    def reset(self) -> StateType:
        if self.state_transform is not None:
            return self.state_transform(self.env.reset())
        return self.env.reset()

    def render(self, mode="rgb_array"):
        if self.can_render:
            # In case of OpenAI, mode can be ['human', 'rgb_array']
            return self.env.render(mode=mode)
        else:
            print("Can't render. Sorry.")  # Yes, this is for haha

    def step(self, actions: ActionType) -> Tuple:
        """
        Each action results in a new state, reward, done flag, and info about env.
        """
        if self.is_discrete:
            actions = int(actions)
        state, reward, done, info = self.env.step(actions)
        if self.state_transform is not None:
            state = self.state_transform(state)
        if self.reward_transform is not None:
            reward = self.reward_transform(reward)
        return (state, reward, done, info)


class MultiAgentUnityTask(MultiAgentTaskType):
    """Based on UnityToGymWrapper from the Unity's ML-Toolkits (permalink [1]).


    At the time of writting the official package doesn't support multi agents.
    Until it's clear why it doesn't support (https://github.com/Unity-Technologies/ml-agents/issues/4120)
    and whether they plan on adding anything, we're keeping this version. When the fog of unknown
    has been blown away, we might consider doing a Pull Request to `ml-agents`.

    [1] https://github.com/Unity-Technologies/ml-agents/blob/3e48be4e1304d8cbbb43d8ffc335f8037cfe6f1d/gym-unity/gym_unity/envs/__init__.py#L27
    """

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

        Parameters
        ----------
            unity_gym_env : GymTask instantiated with UnityEnvironment
        """
        """
        Environment initialization
        :param unity_env: The Unity BaseEnv to be wrapped in the gym. Will be closed when the UnityToGymWrapper closes.
        :param uint8_visual: Return visual observations as uint8 (0-255) matrices instead of float (0.0-1.0).
        :param flatten_branched: If True, turn branched discrete action spaces into a Discrete space rather than
            MultiDiscrete.
        :param allow_multiple_obs: If True, return a list of np.ndarrays as observations with the first elements
            containing the visual observations and the last element containing the array of vector observations.
            If False, returns a single np.ndarray containing either only a single visual observation or the array of
            vector observations.
        :param termination_mode: A string (enum) suggesting when to end an episode. Supports "any", "majority" and "all"
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
        self.agent_prefix = agent_name[:agent_name.index('=')+1]
        self.group_spec = self._env.behavior_specs[agent_name]

        if self._get_n_vis_obs() == 0 and self._get_vec_obs_size() == 0:
            raise UnityGymException(
                "There are no observations provided by the environment."
            )

        if not self._get_n_vis_obs() >= 1 and uint8_visual:
            self.logger.warning(
                "uint8_visual was set to true, but visual observations are not in use. "
                "This setting will not have any effect."
            )
        else:
            self.uint8_visual = uint8_visual
        if (
            self._get_n_vis_obs() + self._get_vec_obs_size() >= 2
            and not self._allow_multiple_obs
        ):
            self.logger.warning(
                "The environment contains multiple observations. "
                "You must define allow_multiple_obs=True to receive them all. "
                "Otherwise, only the first visual observation (or vector observation if"
                "there are no visual observations) will be provided in the observation."
            )

        # Check for number of agents in scene.
        self._env.reset()
        decision_steps, _ = self._env.get_steps(agent_name)
        # self.n_agents = len(decision_steps)  # NOTE: Worked with FoodCollector
        self.n_agents = len(self._env.behavior_specs)
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
                self.logger.warning(
                    "The environment has a non-discrete action space. It will "
                    "not be flattened."
                )
            high = np.ones(self.group_spec.action_shape)
            self._action_space = spaces.Box(-high, high, dtype=np.float32)

        # Set observations space
        list_spaces: List[gym.Space] = []
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
    def reset(self) -> List[StateType]:
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object/list): the initial observation of the
        space.
        """
        self._env.reset()
        states = []
        for agent_id in range(self.n_agents):
            decision_step, _ = self._env.get_steps(self.agent_prefix+str(agent_id))
            self.game_over = False

            res: GymStepResult = self._single_step(decision_step)
            states.append(res[0])  # res contains tuple with `state` on first pos
        return states
        # return res[0]

    def step(self, action: List[Any], agent_id: int) -> GymStepResult:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
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

    def detect_game_over(self, termianl_steps: List[TerminalSteps]) -> bool:
        """Determine whether the episode has finished.

        Expects the `terminal_steps` to contain only steps that terminated. Note that other steps
        are possible in the same iteration.
        This is to keep consistent with Unity's framework but likely will go through refactoring.
        """
        if self.termination_mode == TerminationMode.ANY and len(termianl_steps) > 0:
            return True
        elif self.termination_mode == TerminationMode.MAJORITY and len(termianl_steps) > 0.5 * self.n_agents:
            return True
        elif self.termination_mode == TerminationMode.ALL and len(termianl_steps) == self.n_agents:
            return True
        else:
            return False

    def _single_step(self, info: Union[DecisionSteps, TerminalSteps]) -> GymStepResult:
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

    def _get_vis_obs_shape(self) -> List[Tuple]:
        result: List[Tuple] = []
        for shape in self.group_spec.observation_shapes:
            if len(shape) == 3:
                result.append(shape)
        return result

    def _get_vis_obs_list(
        self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> List[np.ndarray]:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 4:
                result.append(obs)
        return result

    def _get_vector_obs(
        self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> np.ndarray:
        result: List[np.ndarray] = []
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
    def reward_range(self) -> Tuple[float, float]:
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
