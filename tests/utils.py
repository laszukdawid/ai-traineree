import numpy as np


def generate_sample_SARS(iterations, obs_size: int = 4, action_size: int = 2, dict_type=False):
    def state_fn():
        return np.random.random(obs_size)

    def action_fn():
        return np.random.random(action_size)

    def reward_fn():
        return float(np.random.random() - 0.5)

    def done_fn():
        return np.random.random() > 0.5

    state = state_fn()

    for _ in range(iterations):
        next_state = state_fn()
        if dict_type:
            yield dict(
                state=list(state),
                action=list(action_fn()),
                reward=[reward_fn()],
                next_state=list(next_state),
                done=[bool(done_fn())],
            )
        else:
            yield (list(state), list(action_fn()), reward_fn(), list(next_state), bool(done_fn()))
        state = next_state
