import pytest

from ai_traineree.buffers.nstep import NStepBuffer
from ai_traineree.types.experience import Experience
from tests.buffers.test_buffer_factory import generate_sample_SARS


def populate_buffer(buffer, num_samples):
    for state, action, reward, next_state, done in generate_sample_SARS(num_samples):
        buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)
    return buffer


def test_nstep_buffer_add_sample():
    # Assign
    buffer = NStepBuffer(2, gamma=1.0)

    # Act
    sars = next(generate_sample_SARS(1, dict_type=True))
    buffer.add(**sars)

    # Assert
    assert len(buffer) == 1
    assert list(buffer.data) == [Experience(**sars)]


def test_nstep_buffer_add_many_samples():
    # Assign
    buffer_size = 4
    gamma = 1.0
    buffer = NStepBuffer(n_steps=buffer_size, gamma=gamma)
    populate_buffer(buffer, 20)  # in-place
    last_samples = [sars for sars in generate_sample_SARS(buffer_size, dict_type=True)]
    last_rewards = [s["reward"][0] for s in last_samples]

    # Act
    for sample in last_samples:
        sample["done"] = [False]  # Make sure all samples are counted
        buffer.add(**sample)

    # Assert
    assert len(buffer) == buffer_size
    for expected_len in range(buffer_size)[::-1]:
        sample = buffer.get()
        assert len(buffer) == expected_len
        assert sample.reward[0] == pytest.approx(sum(last_rewards[-expected_len - 1 :]), abs=1e-9)


def test_nstep_buffer_add_many_samples_discounted():
    # Assign
    buffer_size = 4
    gamma = 0.9
    buffer = NStepBuffer(n_steps=buffer_size, gamma=gamma)
    populate_buffer(buffer, 20)  # in-place
    last_samples = [sars for sars in generate_sample_SARS(4, dict_type=True)]
    last_rewards = [s["reward"][0] for s in last_samples]

    # Act
    for sample in last_samples:
        sample["done"] = [False]  # Make sure all samples are counted
        buffer.add(**sample)

    # Assert
    assert len(buffer) == buffer_size
    for expected_len in range(buffer_size)[::-1]:
        sample = buffer.get()
        discounted_reward = sum([r * gamma**idx for (idx, r) in enumerate(last_rewards[-expected_len - 1 :])])
        assert len(buffer) == expected_len
        assert sample.reward[0] == pytest.approx(discounted_reward, abs=1e-9)


def test_nstep_buffer_add_many_samples_discounted_terminate():
    # Assign
    buffer_size = 4
    gamma = 0.9
    buffer = NStepBuffer(n_steps=buffer_size, gamma=gamma)
    populate_buffer(buffer, 20)  # in-place
    last_samples = [sars for sars in generate_sample_SARS(4, dict_type=True)]

    expected_rewards = []
    for idx, sample in enumerate(last_samples):
        expected_rewards.append(sample["reward"][0])
        for iidx, sample in enumerate(last_samples[idx + 1 :]):
            if any(sample["done"]):
                break
            expected_rewards[-1] += gamma ** (iidx + 1) * sample["reward"][0]

    # Act
    for sample in last_samples:
        buffer.add(**sample)

    # Assert
    assert len(buffer) == buffer_size
    for idx, expected_len in enumerate(range(buffer_size)[::-1]):
        sample = buffer.get()
        assert len(buffer) == expected_len
        assert sample.reward[0] == expected_rewards[idx], f"{sample}"


def test_nstep_buffer_clear():
    # Assign
    buffer = NStepBuffer(n_steps=5, gamma=1.0)
    populate_buffer(buffer, 10)  # in-place

    # Act & assert
    assert len(buffer) == 5

    buffer.clear()
    assert len(buffer) == 0


def test_nstep_buffer_get_state_without_data():
    # Assign
    buffer = NStepBuffer(n_steps=5, gamma=1.0)

    # Act
    state = buffer.get_state()

    # Assert
    assert state.type == NStepBuffer.type
    assert state.buffer_size == 5
    assert state.batch_size == 1
    assert state.data is None


def test_nstep_buffer_get_state_with_data():
    # Assign
    buffer = NStepBuffer(n_steps=5, gamma=1.0)
    populate_buffer(buffer, 10)  # in-place

    # Act
    state = buffer.get_state()

    # Assert
    assert state.type == NStepBuffer.type
    assert state.buffer_size == 5
    assert state.batch_size == 1
    assert len(state.data) == state.buffer_size


def test_nstep_buffer_from_state_without_data():
    # Assign
    buffer_size, gamma = 5, 0.9
    buffer = NStepBuffer(n_steps=buffer_size, gamma=gamma)
    state = buffer.get_state()

    # Act
    new_buffer = NStepBuffer.from_state(state)

    # Assert
    assert new_buffer.type == NStepBuffer.type
    assert new_buffer.gamma == gamma
    assert new_buffer.buffer_size == state.buffer_size == buffer.n_steps
    assert new_buffer.batch_size == state.batch_size == buffer.batch_size == 1
    assert len(new_buffer.data) == 0


def test_nstep_buffer_from_state_with_data():
    # Assign
    buffer_size = 5
    buffer = NStepBuffer(n_steps=buffer_size, gamma=1.0)
    buffer = populate_buffer(buffer, 10)  # in-place
    last_samples = [sars for sars in generate_sample_SARS(buffer_size, dict_type=True)]
    for sample in last_samples:
        buffer.add(**sample)
    state = buffer.get_state()

    # Act
    new_buffer = NStepBuffer.from_state(state)

    # Assert
    assert new_buffer.type == NStepBuffer.type
    assert new_buffer.buffer_size == state.buffer_size == buffer.n_steps
    assert new_buffer.batch_size == state.batch_size == buffer.batch_size == 1
    assert len(new_buffer.data) == state.buffer_size

    for sample in last_samples:
        assert Experience(**sample) in new_buffer.data


def test_per_from_state_wrong_type():
    # Assign
    buffer = NStepBuffer(n_steps=5, gamma=1.0)
    state = buffer.get_state()
    state.type = "WrongType"

    # Act
    with pytest.raises(ValueError):
        NStepBuffer.from_state(state=state)


def test_per_from_state_wrong_batch_size():
    # Assign
    buffer = NStepBuffer(n_steps=5, gamma=1.0)
    state = buffer.get_state()
    state.batch_size = 5

    # Act
    with pytest.raises(ValueError):
        NStepBuffer.from_state(state=state)
