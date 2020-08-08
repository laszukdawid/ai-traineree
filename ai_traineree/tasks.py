from ai_traineree.types import StateType, TaskType


class GymTask(TaskType):
    def __init__(self, env, env_name: str, can_render=True):
        self.name = env_name
        self.env = env
        self.can_render = can_render
        self.is_discrete = "Discrete" in str(type(env.action_space))

        self.state_size = env.observation_space.shape[0]
        self.action_size = self.__determine_action_size(env.action_space)

    @staticmethod
    def __determine_action_size(action_space):
        if "Discrete" in str(type(action_space)):
            return action_space.n
        else:
            return sum(action_space.shape)

    def reset(self) -> StateType:
        return self.env.reset()

    def render(self):
        if self.can_render:
            self.env.render()
        else:
            print("Can't render. Sorry.")  # Yes, this is for haha

    def step(self, actions):
        if self.is_discrete:
            actions = int(actions)
        return self.env.step(actions)
