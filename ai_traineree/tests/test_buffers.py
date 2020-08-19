import numpy as np

from ai_traineree.buffers import Experience, PERBuffer, ReplayBuffer


def generate_sample_SARS(state_size: int=4, action_size: int=2):
    states = np.random.random((2, state_size))
    action = np.random.random(action_size)
    reward = float(np.random.random() - 0.5)
    done = np.random.random() > 0.5
    return (list(states[0]), list(action), reward, list(states[1]), bool(done))


def test_buffer_size():
    # Assign
    buffer_size = 10
    buffer = ReplayBuffer(batch_size=5, buffer_size=buffer_size)

    # Act
    for _ in range(buffer_size+2):
        (state, action, reward, next_state, done) = generate_sample_SARS()
        buffer.add_sars(state=state, action=action, reward=reward, next_state=next_state, done=done)

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


def test_buffer_add():
    # Assign
    buffer = ReplayBuffer(batch_size=5, buffer_size=5)

    # Act
    assert len(buffer) == 0
    (state, actions, reward, next_state, done) = generate_sample_SARS()
    buffer.add_sars(state=state, action=actions, reward=reward, next_state=next_state, done=done)

    # Assert
    assert len(buffer) == 1


def test_buffer_sample():
    # Assign
    batch_size = 5
    buffer = ReplayBuffer(batch_size=batch_size, buffer_size=10)

    # Act
    for _ in range(20):
        (state, actions, reward, next_state, done) = generate_sample_SARS()
        buffer.add_sars(state=state, action=actions, reward=reward, next_state=next_state, done=done)

    # Assert
    (states, actions, rewards, next_states, dones) = buffer.sample_sars()
    assert len(states) == batch_size
    assert len(actions) == batch_size
    assert len(rewards) == batch_size
    assert len(next_states) == batch_size
    assert len(dones) == batch_size


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
    for _ in range(batch_size):
        assert per_buffer.sample_list() is None
        per_buffer.add(priority=0.1, reward=0.1)

    assert per_buffer.sample_list() is not None


def test_per_buffer_add_one_sample_one():
    # Assign
    per_buffer = PERBuffer(1, 20)

    # Act
    per_buffer.add(priority=0.5, state=range(5))

    # Assert
    raw_samples = per_buffer.sample_list()
    assert raw_samples is not None
    experience = raw_samples[0]
    assert experience.state == range(5)
    assert experience.weight == 1.  # max scale
    assert experience.index == 0


def test_per_buffer_add_two_sample_two_beta():
    # Assign
    per_buffer = PERBuffer(2, 20)

    # Act
    per_buffer.add(state=range(5), priority=0.9)
    per_buffer.add(state=range(3, 8), priority=0.1)

    # Assert
    experiences = per_buffer.sample_list(beta=0.6)
    assert experiences is not None
    for experience in experiences:
        if experience.index == 0:
            assert experience.state == range(5)
            assert 0.936 < experience.weight < 0.937
        else:
            assert experience.state == range(3, 8)
            assert experience.weight == 1.


def test_per_buffer_sample():
    # Assign
    buffer_size = 5
    per_buffer = PERBuffer(buffer_size)

    # Act
    for priority in range(buffer_size):
        state = np.arange(priority, priority+10)
        per_buffer.add(priority=priority+0.01, state=state)

    # Assert
    experiences = per_buffer.sample_list()
    assert experiences is not None
    assert len(experiences) == buffer_size
    zipped_exp = [(exp.state, exp.reward, exp.weight, exp.index) for exp in experiences]
    states, rewards, weights, indices = zip(*zipped_exp)
    assert len(weights) == len(indices) == buffer_size
    assert all([s is not None for s in states])
    assert all([r is None for r in rewards])


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
    experiences = per_buffer.sample_list(beta=0.5)
    assert experiences is not None
    assert sum([exp.weight for exp in experiences]) < batch_size

    per_buffer.priority_update(indices=range(buffer_size), priorities=np.ones(buffer_size))
    experiences = per_buffer.sample_list(beta=0.9)
    assert experiences is not None
    weights = [exp.weight for exp in experiences]
    assert sum(weights) == batch_size
    assert all([w == 1 for w in weights])


def test_per_buffer_reset_alpha():
    # Assign
    per_buffer = PERBuffer(10, 10, alpha=0.1)
    for _ in range(30):
        per_buffer.add(reward=np.random.randint(0, 1e5), priority=np.random.random())

    # Act
    old_experiences = per_buffer.sample_list()
    per_buffer.reset_alpha(0.5)
    new_experiences = per_buffer.sample_list()

    # Assert
    assert old_experiences is not None and new_experiences is not None
    sorted_new_experiences = sorted(new_experiences, key=lambda k: k.index)
    sorted_old_experiences = sorted(old_experiences, key=lambda k: k.index)
    for (new_sample, old_sample) in zip(sorted_new_experiences, sorted_old_experiences):
        assert new_sample.index == old_sample.index
        assert new_sample.weight != old_sample.weight
        assert new_sample.reward == old_sample.reward
