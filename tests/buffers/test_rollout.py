import pytest

from ai_traineree.buffers import Experience, RolloutBuffer
from tests.utils import generate_sample_SARS


def populate_buffer(buffer, num_samples):
    for (state, action, reward, next_state, done) in generate_sample_SARS(num_samples):
        buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)
    return buffer


def test_rollout_buffer_length():
    # Assign
    buffer_size = 10
    buffer = RolloutBuffer(batch_size=5, buffer_size=buffer_size)

    # Act
    for (state, action, reward, next_state, done) in generate_sample_SARS(buffer_size + 1):
        buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

    # Assert
    assert len(buffer) == buffer_size


def test_rollout_buffer_sample_batch_equal_buffer():
    # Assign
    buffer_size = batch_size = 20
    buffer = RolloutBuffer(batch_size=batch_size, buffer_size=buffer_size)

    # Act
    for (state, action, reward, next_state, done) in generate_sample_SARS(buffer_size + 1):
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
    for (state, action, reward, next_state, done) in generate_sample_SARS(buffer_size + 1):
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
        rewards = samples["reward"]
        if idx != 5:
            assert len(rewards) == batch_size
            assert rewards == list(range(idx * 10, (idx + 1) * 10))
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
        rewards = samples["reward"]
        assert rewards == list(range(idx * 10, (idx + 1) * 10))

    # Second pass
    for idx, samples in enumerate(buffer.sample()):
        num_samples += 1
        rewards = samples["reward"]
        assert rewards == list(range(idx * 10, (idx + 1) * 10))

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
        rewards = samples["reward"]
        assert rewards == list(range(idx * 10, (idx + 1) * 10))

    buffer.clear()
    assert len(buffer) == 0


def test_rollout_buffer_get_state_without_data():
    # Assign
    buffer = RolloutBuffer(batch_size=5, buffer_size=20)

    # Act
    state = buffer.get_state()

    # Assert
    assert state.type == RolloutBuffer.type
    assert state.buffer_size == 20
    assert state.batch_size == 5
    assert state.data is None


def test_rollout_buffer_get_state_with_data():
    # Assign
    buffer = RolloutBuffer(batch_size=5, buffer_size=20)
    sample_experience = Experience(state=[0, 1], action=[0], reward=0)
    for _ in range(25):
        buffer.add(**sample_experience.data)

    # Act
    state = buffer.get_state()

    # Assert
    assert state.type == RolloutBuffer.type
    assert state.buffer_size == 20
    assert state.batch_size == 5
    assert state.data is not None
    assert len(state.data) == 20

    for data in state.data:
        assert data == sample_experience


def test_rollout_buffer_from_state_without_data():
    # Assign
    buffer = RolloutBuffer(batch_size=5, buffer_size=20)
    state = buffer.get_state()

    # Act
    new_buffer = RolloutBuffer.from_state(state=state)

    # Assert
    assert new_buffer == buffer
    assert new_buffer.buffer_size == state.buffer_size
    assert new_buffer.batch_size == state.batch_size
    assert len(new_buffer.data) == 0


def test_rollout_buffer_from_state_with_data():
    # Assign
    buffer = RolloutBuffer(batch_size=5, buffer_size=20)
    buffer = populate_buffer(buffer, 30)
    state = buffer.get_state()

    # Act
    new_buffer = RolloutBuffer.from_state(state=state)

    # Assert
    assert new_buffer == buffer
    assert new_buffer.buffer_size == state.buffer_size
    assert new_buffer.batch_size == state.batch_size
    assert new_buffer.data == state.data
    assert len(buffer.data) == state.buffer_size


def test_rollout_buffer_from_state_wrong_type():
    # Assign
    buffer = RolloutBuffer(batch_size=5, buffer_size=20)
    state = buffer.get_state()
    state.type = "WrongType"

    # Act
    with pytest.raises(ValueError):
        RolloutBuffer.from_state(state=state)
