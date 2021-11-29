import pytest

from ai_traineree.buffers.buffer_factory import BufferFactory
from ai_traineree.buffers.nstep import NStepBuffer
from ai_traineree.buffers.per import PERBuffer
from ai_traineree.buffers.replay import ReplayBuffer
from ai_traineree.buffers.rollout import RolloutBuffer
from ai_traineree.types.experience import Experience
from tests.buffers.test_replay import generate_sample_SARS


def populate_buffer(buffer, num_samples):
    for (state, action, reward, next_state, done) in generate_sample_SARS(num_samples):
        buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)
    return buffer


def test_factory_nstep_buffer_from_state_without_data():
    # Assign
    buffer_size, gamma = 5, 0.9
    buffer = NStepBuffer(n_steps=buffer_size, gamma=gamma)
    state = buffer.get_state()

    # Act
    new_buffer = BufferFactory.from_state(state)

    # Assert
    assert new_buffer.type == NStepBuffer.type
    assert new_buffer.gamma == gamma
    assert new_buffer.buffer_size == state.buffer_size == buffer.n_steps
    assert new_buffer.batch_size == state.batch_size == buffer.batch_size == 1
    assert len(new_buffer.data) == 0


def test_factory_nstep_buffer_from_state_with_data():
    # Assign
    buffer_size = 5
    buffer = NStepBuffer(n_steps=buffer_size, gamma=1.0)
    buffer = populate_buffer(buffer, 10)  # in-place
    last_samples = [sars for sars in generate_sample_SARS(buffer_size, dict_type=True)]
    for sample in last_samples:
        buffer.add(**sample)
    state = buffer.get_state()

    # Act
    new_buffer = BufferFactory.from_state(state)

    # Assert
    assert new_buffer.type == NStepBuffer.type
    assert new_buffer.buffer_size == state.buffer_size == buffer.n_steps
    assert new_buffer.batch_size == state.batch_size == buffer.batch_size == 1
    assert len(new_buffer.data) == state.buffer_size

    for sample in last_samples:
        assert Experience(**sample) in new_buffer.data


def test_factory_per_from_state_without_data():
    # Assign
    buffer = PERBuffer(batch_size=5, buffer_size=20)
    state = buffer.get_state()

    # Act
    new_buffer = BufferFactory.from_state(state=state)

    # Assert
    assert new_buffer == buffer
    assert new_buffer.buffer_size == state.buffer_size
    assert new_buffer.batch_size == state.batch_size
    assert new_buffer.data == []


def test_factory_per_from_state_with_data():
    # Assign
    buffer = PERBuffer(batch_size=5, buffer_size=20)
    buffer = populate_buffer(buffer, 30)
    state = buffer.get_state()

    # Act
    new_buffer = BufferFactory.from_state(state=state)

    # Assert
    assert new_buffer == buffer
    assert new_buffer.buffer_size == state.buffer_size
    assert new_buffer.batch_size == state.batch_size
    assert new_buffer.data == state.data
    assert len(buffer.data) == state.buffer_size


def test_factory_per_from_state_wrong_type():
    # Assign
    buffer = PERBuffer(batch_size=5, buffer_size=20)
    state = buffer.get_state()
    state.type = "WrongType"

    # Act
    with pytest.raises(ValueError):
        BufferFactory.from_state(state=state)


def test_factory_replay_buffer_from_state_without_data():
    # Assign
    buffer = ReplayBuffer(batch_size=5, buffer_size=20)
    state = buffer.get_state()

    # Act
    new_buffer = BufferFactory.from_state(state=state)

    # Assert
    assert new_buffer == buffer
    assert new_buffer.buffer_size == state.buffer_size
    assert new_buffer.batch_size == state.batch_size
    assert new_buffer.data == []


def test_factory_replay_buffer_from_state_with_data():
    # Assign
    buffer = ReplayBuffer(batch_size=5, buffer_size=20)
    buffer = populate_buffer(buffer, 30)
    state = buffer.get_state()

    # Act
    new_buffer = BufferFactory.from_state(state=state)

    # Assert
    assert new_buffer == buffer
    assert new_buffer.buffer_size == state.buffer_size
    assert new_buffer.batch_size == state.batch_size
    assert new_buffer.data == state.data
    assert len(buffer.data) == state.buffer_size


def test_factory_replay_buffer_from_state_wrong_type():
    # Assign
    buffer = ReplayBuffer(batch_size=5, buffer_size=20)
    state = buffer.get_state()
    state.type = "WrongType"

    # Act
    with pytest.raises(ValueError):
        BufferFactory.from_state(state=state)


def test_factory_rollout_buffer_from_state_without_data():
    # Assign
    buffer = RolloutBuffer(batch_size=5, buffer_size=20)
    state = buffer.get_state()

    # Act
    new_buffer = BufferFactory.from_state(state=state)

    # Assert
    assert new_buffer == buffer
    assert new_buffer.buffer_size == state.buffer_size
    assert new_buffer.batch_size == state.batch_size
    assert len(new_buffer.data) == 0


def test_factory_rollout_buffer_from_state_with_data():
    # Assign
    buffer = RolloutBuffer(batch_size=5, buffer_size=20)
    buffer = populate_buffer(buffer, 30)
    state = buffer.get_state()

    # Act
    new_buffer = BufferFactory.from_state(state=state)

    # Assert
    assert new_buffer == buffer
    assert new_buffer.buffer_size == state.buffer_size
    assert new_buffer.batch_size == state.batch_size
    assert new_buffer.data == state.data
    assert len(buffer.data) == state.buffer_size


def test_factory_rollout_buffer_from_state_wrong_type():
    # Assign
    buffer = RolloutBuffer(batch_size=5, buffer_size=20)
    state = buffer.get_state()
    state.type = "WrongType"

    # Act
    with pytest.raises(ValueError):
        BufferFactory.from_state(state=state)
