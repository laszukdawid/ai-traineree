import copy
import numpy as np

from ai_traineree.buffers import Experience, PERBuffer, ReplayBuffer, RolloutBuffer
from typing import Dict, List


def generate_sample_SARS(iterations, state_size: int=4, action_size: int=2, dict_type=False):
    state_fn = lambda: np.random.random(state_size)
    action_fn = lambda: np.random.random(action_size)
    reward_fn = lambda: float(np.random.random() - 0.5)
    done_fn = lambda: np.random.random() > 0.5
    state = state_fn()

    for _ in range(iterations):
        next_state = state_fn()
        if dict_type:
            yield dict(
                state=list(state), action=list(action_fn()), reward=[reward_fn()], next_state=list(next_state), done=[bool(done_fn())]
            )
        else:
            yield (list(state), list(action_fn()), reward_fn(), list(next_state), bool(done_fn()))
        state = next_state


def test_buffer_size():
    # Assign
    buffer_size = 10
    buffer = ReplayBuffer(batch_size=5, buffer_size=buffer_size)

    # Act
    for (state, action, reward, next_state, done) in generate_sample_SARS(buffer_size+1):
        buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

    # Assert
    assert len(buffer) == buffer_size


def test_experience_init():
    # Assign
    state = np.random.random(10)
    action = np.random.random(2)
    reward = 3
    next_state = np.random.random(10)
    done = False

    exp = Experience(state=state, action=action, reward=reward)
    assert all(exp.state == state)
    assert all(exp.action == action)
    assert exp.reward == reward
    assert exp.next_state is None
    assert exp.done is None

    exp = Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
    assert all(exp.state == state)
    assert all(exp.action == action)
    assert exp.reward == reward
    assert all(exp.next_state == next_state)
    assert exp.done == done


def test_replay_buffer_add():
    # Assign
    buffer = ReplayBuffer(batch_size=5, buffer_size=5)

    # Act
    assert len(buffer) == 0
    for sars in generate_sample_SARS(1, dict_type=True):
        buffer.add(**sars)

    # Assert
    assert len(buffer) == 1


def test_replay_buffer_sample():
    # Assign
    batch_size = 5
    buffer = ReplayBuffer(batch_size=batch_size, buffer_size=10)

    # Act
    for (state, actions, reward, next_state, done) in generate_sample_SARS(20):
        buffer.add(state=state, action=actions, reward=reward, next_state=next_state, done=done)

    # Assert
    samples = buffer.sample()
    # (states, actions, rewards, next_states, dones)
    assert len(samples["state"]) == batch_size
    assert len(samples["action"]) == batch_size
    assert len(samples["reward"]) == batch_size
    assert len(samples["next_state"]) == batch_size
    assert len(samples["done"]) == batch_size


def test_replay_buffer_seed():
    # Assign
    batch_size = 4
    buffer_0 = ReplayBuffer(batch_size)
    buffer_1 = ReplayBuffer(batch_size, seed=32167)
    buffer_2 = ReplayBuffer(batch_size, seed=32167)

    # Act
    for sars in generate_sample_SARS(400, dict_type=True):
        buffer_0.add(**copy.deepcopy(sars))
        buffer_1.add(**copy.deepcopy(sars))
        buffer_2.add(**copy.deepcopy(sars))

    # Assert
    for _ in range(10):
        samples_0 = buffer_0.sample()
        samples_1 = buffer_1.sample()
        samples_2 = buffer_2.sample()

        assert samples_0 != samples_1
        assert samples_0 != samples_2
        assert samples_1 == samples_2


def test_per_buffer_len():
    # Assign
    buffer_size = 10
    per_buffer = PERBuffer(5, buffer_size)

    # Act & Assert
    for sample_num in range(buffer_size+2):
        assert len(per_buffer) == min(sample_num, buffer_size)
        per_buffer.add(priority=1, state=1)


def test_per_buffer_too_few_samples():
    # Assign
    batch_size = 5
    per_buffer = PERBuffer(batch_size, 10)

    # Act & Assert
    for _ in range(batch_size - 1):
        per_buffer.add(priority=0.1, reward=0.1)
        assert per_buffer.sample() is None

    per_buffer.add(priority=0.1, reward=0.1)
    assert len(per_buffer.sample()['reward']) == 5


def test_per_buffer_add_one_sample_one():
    # Assign
    per_buffer = PERBuffer(1, 20)

    # Act
    per_buffer.add(priority=0.5, state=range(5))

    # Assert
    samples = per_buffer.sample()
    assert samples is not None
    assert samples['state'] == [range(5)]
    assert samples['weight'] == [1.]  # max scale
    assert samples['index'] == [0]


def test_per_buffer_add_two_sample_two_beta():
    # Assign
    per_buffer = PERBuffer(2, 20, 0.4)

    # Act
    per_buffer.add(state=range(5), priority=0.9)
    per_buffer.add(state=range(3, 8), priority=0.1)

    # Assert
    experiences = per_buffer.sample(beta=0.6)
    assert experiences is not None
    for (state, weight) in zip(experiences['state'], experiences['weight']):
        if weight == 1:
            assert state == range(3, 8)
        else:
            assert 0.6421 < weight < 0.6422
            assert state == range(5)


def test_per_buffer_sample():
    # Assign
    buffer_size = 5
    per_buffer = PERBuffer(buffer_size)

    # Act
    for priority in range(buffer_size):
        state = np.arange(priority, priority+10)
        per_buffer.add(priority=priority+0.01, state=state)

    # Assert
    experiences = per_buffer.sample()
    assert experiences is not None
    state = experiences['state']
    reward = experiences['reward']
    weight = experiences['weight']
    index = experiences['index']
    assert len(state) == len(reward) == len(weight) == len(index) == buffer_size
    assert all([s is not None for s in state])
    assert all([r is None for r in reward])


def test_per_buffer_priority_update():
    """Update all priorities to the same value makes them all to be 1."""
    # Assign
    batch_size = 5
    buffer_size = 10
    per_buffer = PERBuffer(batch_size, buffer_size)
    for _ in range(2*buffer_size):  # Make sure we fill the whole buffer
        per_buffer.add(priority=np.random.randint(10), state=np.random.random(10))
    per_buffer.add(priority=100, state=np.random.random(10))  # Make sure there's one highest

    # Act & Assert
    experiences = per_buffer.sample(beta=0.5)
    assert experiences is not None
    assert sum(experiences['weight']) < batch_size
    # assert sum([weight for exp in experiences]) < batch_size

    per_buffer.priority_update(indices=range(buffer_size), priorities=np.ones(buffer_size))
    experiences = per_buffer.sample(beta=0.9)
    assert experiences is not None
    # weights = [exp.weight for exp in experiences]

    assert sum(experiences['weight']) == batch_size
    assert all([w == 1 for w in experiences['weight']])


def test_per_buffer_reset_alpha():
    # Assign
    per_buffer = PERBuffer(10, 10, alpha=0.1)
    for _ in range(30):
        per_buffer.add(reward=np.random.randint(0, 1e5), priority=np.random.random())

    # Act
    old_experiences = per_buffer.sample()
    per_buffer.reset_alpha(0.5)
    new_experiences = per_buffer.sample()

    # Assert
    assert old_experiences is not None and new_experiences is not None
    old_index, new_index = np.array(old_experiences['index']), np.array(new_experiences['index'])
    old_weight, new_weight = np.array(old_experiences['weight']), np.array(new_experiences['weight'])
    old_reward, new_reward = np.array(old_experiences['reward']), np.array(new_experiences['reward'])
    old_sort, new_sort = np.argsort(old_index), np.argsort(new_index)
    assert all([i1 == i2 for (i1, i2) in zip(old_index[old_sort], new_index[new_sort])])
    assert all([w1 != w2 for (w1, w2) in zip(old_weight[old_sort], new_weight[new_sort])])
    assert all([r1 == r2 for (r1, r2) in zip(old_reward[old_sort], new_reward[new_sort])])


def test_replay_buffer_dump():
    import torch
    # Assign
    filled_buffer = 8
    prop_keys = ["state", "action", "reward", "next_state"]
    buffer = ReplayBuffer(batch_size=5, buffer_size=10)
    for sars in generate_sample_SARS(filled_buffer):
        buffer.add(state=torch.tensor(sars[0]), reward=sars[1], action=[sars[2]], next_state=torch.tensor(sars[3]), dones=sars[4])

    # Act
    dump: List[Dict[str, List]] = list(buffer.dump_buffer())

    # Assert
    assert all([len(dump) == filled_buffer])
    assert all([key in dump[0] for key in prop_keys])


def test_replay_buffer_dump_serializable():
    import json
    import torch
    # Assign
    filled_buffer = 8
    buffer = ReplayBuffer(batch_size=5, buffer_size=10)

    for sars in generate_sample_SARS(filled_buffer, dict_type=True):
        sars['state'] = torch.tensor(sars['state'])
        sars['next_state'] = torch.tensor(sars['next_state'])
        buffer.add(**sars)

    # Act
    dump = list(buffer.dump_buffer(serialize=True))

    # Assert
    ser_dump = json.dumps(dump)
    assert isinstance(ser_dump, str)
    assert json.loads(ser_dump) == dump


def test_replay_buffer_load_json_dump():
    # Assign
    prop_keys = ["state", "action", "reward", "next_state", "done"]
    buffer = ReplayBuffer(batch_size=20, buffer_size=20)
    ser_buffer = []
    for sars in generate_sample_SARS(10, dict_type=True):
        ser_buffer.append(sars)

    # Act
    buffer.load_buffer(ser_buffer)

    # Assert
    samples = buffer.exp
    assert len(buffer) == 10
    assert len(samples) == 10
    for sample in samples:
        assert all([hasattr(sample, key) for key in prop_keys])
        assert all([isinstance(getattr(sample, key), list) for key in prop_keys])


def test_priority_buffer_dump_serializable():
    import json
    import torch
    # Assign
    filled_buffer = 8
    buffer = PERBuffer(batch_size=5, buffer_size=10)
    for sars in generate_sample_SARS(filled_buffer):
        buffer.add(state=torch.tensor(sars[0]), reward=sars[1], action=[sars[2]], next_state=torch.tensor(sars[3]), dones=sars[4])

    # Act
    dump = list(buffer.dump_buffer(serialize=True))

    # Assert
    ser_dump = json.dumps(dump)
    assert isinstance(ser_dump, str)
    assert json.loads(ser_dump) == dump


def test_priority_buffer_load_json_dump():
    # Assign
    prop_keys = ["state", "action", "reward", "next_state", "done"]
    buffer = PERBuffer(batch_size=10, buffer_size=20)
    ser_buffer = []
    for sars in generate_sample_SARS(10, dict_type=True):
        ser_buffer.append(sars)

    # Act
    buffer.load_buffer(ser_buffer)

    # Assert
    samples = buffer._sample_list()
    assert len(buffer) == 10
    assert len(samples) == 10
    for sample in samples:
        assert all([hasattr(sample, key) for key in prop_keys])
        assert all([isinstance(getattr(sample, key), list) for key in prop_keys])


def test_per_buffer_seed():
    # Assign
    batch_size = 4
    buffer_0 = PERBuffer(batch_size)
    buffer_1 = PERBuffer(batch_size, seed=32167)
    buffer_2 = PERBuffer(batch_size, seed=32167)

    # Act
    for sars in generate_sample_SARS(400, dict_type=True):
        buffer_0.add(**copy.deepcopy(sars))
        buffer_1.add(**copy.deepcopy(sars))
        buffer_2.add(**copy.deepcopy(sars))

    # Assert
    for _ in range(10):
        samples_0 = buffer_0.sample()
        samples_1 = buffer_1.sample()
        samples_2 = buffer_2.sample()

        assert samples_0 != samples_1
        assert samples_0 != samples_2
        assert samples_1 == samples_2


def test_rollout_buffer_length():
    # Assign
    buffer_size = 10
    buffer = RolloutBuffer(batch_size=5, buffer_size=buffer_size)

    # Act
    for (state, action, reward, next_state, done) in generate_sample_SARS(buffer_size+1):
        buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

    # Assert
    assert len(buffer) == buffer_size


def test_rollout_buffer_sample_batch_equal_buffer():
    # Assign
    buffer_size = batch_size = 20
    buffer = RolloutBuffer(batch_size=batch_size, buffer_size=buffer_size)

    # Act
    for (state, action, reward, next_state, done) in generate_sample_SARS(buffer_size+1):
        buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

    # Assert
    num_samples = 0
    for samples in buffer.sample():
        num_samples += 1
        for value in samples.values():
            assert len(value) == batch_size
    assert num_samples == 1


def test_rollout_buffer_size_multiple_of_minibatch():
    # Assign
    batch_size = 10
    buffer_size = 50
    buffer = RolloutBuffer(batch_size=batch_size, buffer_size=buffer_size)

    # Act
    for (state, action, reward, next_state, done) in generate_sample_SARS(buffer_size+1):
        buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

    # Assert
    num_samples = 0
    for samples in buffer.sample():
        num_samples += 1
        for value in samples.values():
            assert len(value) == batch_size
    assert num_samples == 5  # buffer_size / batch_size


def test_rollout_buffer_size_not_multiple_of_minibatch():
    # Assign
    batch_size = 10
    buffer_size = 55
    buffer = RolloutBuffer(batch_size=batch_size, buffer_size=buffer_size)

    # Act
    reward = -1
    for (state, action, _, next_state, done) in generate_sample_SARS(buffer_size):
        reward += 1
        buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

    # Assert
    num_samples = 0
    for idx, samples in enumerate(buffer.sample()):
        num_samples += 1
        rewards = samples['reward']
        if idx != 5:
            assert len(rewards) == batch_size
            assert rewards == list(range(idx*10, (idx+1)*10))
        else:
            assert len(rewards) == 5
            assert rewards == [50, 51, 52, 53, 54]
    assert num_samples == 6  # ceil(buffer_size / batch_size)


def test_rollout_buffer_travers_buffer_twice():
    # Assign
    batch_size = 10
    buffer_size = 30
    buffer = RolloutBuffer(batch_size=batch_size, buffer_size=buffer_size)

    # Act
    reward = -1
    for (state, action, _, next_state, done) in generate_sample_SARS(buffer_size):
        reward += 1
        buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

    # Assert
    num_samples = 0

    # First pass
    for idx, samples in enumerate(buffer.sample()):
        num_samples += 1
        rewards = samples['reward']
        assert rewards == list(range(idx*10, (idx+1)*10))

    # Second pass
    for idx, samples in enumerate(buffer.sample()):
        num_samples += 1
        rewards = samples['reward']
        assert rewards == list(range(idx*10, (idx+1)*10))

    assert num_samples == 6  # 2 * ceil(buffer_size / batch_size)


def test_rollout_buffer_clear_buffer():
    # Assign
    batch_size = 10
    buffer_size = 30
    buffer = RolloutBuffer(batch_size=batch_size, buffer_size=buffer_size)

    # Act
    reward = -1
    for (state, action, _, next_state, done) in generate_sample_SARS(buffer_size):
        reward += 1
        buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

    # Assert
    for idx, samples in enumerate(buffer.sample()):
        rewards = samples['reward']
        assert rewards == list(range(idx*10, (idx+1)*10))

    buffer.clear()
    assert len(buffer) == 0
