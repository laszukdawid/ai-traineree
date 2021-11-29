import copy

import pytest

from ai_traineree.buffers.replay import ReplayBuffer
from ai_traineree.types.experience import Experience
from ai_traineree.types.state import BufferState
from tests.utils import generate_sample_SARS


def populate_buffer(buffer, num_samples):
    for (state, action, reward, next_state, done) in generate_sample_SARS(num_samples):
        buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)
    return buffer


def test_buffer_size():
    # Assign
    buffer_size = 10
    buffer = ReplayBuffer(batch_size=5, buffer_size=buffer_size)

    # Act
    for (state, action, reward, next_state, done) in generate_sample_SARS(buffer_size + 1):
        buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

    # Assert
    assert len(buffer) == buffer_size


def test_replay_buffer_add():
    # Assign
    buffer = ReplayBuffer(batch_size=5, buffer_size=5)

    # Act
    assert len(buffer) == 0
    for sars in generate_sample_SARS(1, dict_type=True):
        buffer.add(**sars)

    # Assert
    assert len(buffer) == 1


def test_replay_buffer_clear():
    # Assign
    buffer = ReplayBuffer(batch_size=5, buffer_size=5)

    # Act # Assert
    assert len(buffer) == 0
    for sars in generate_sample_SARS(10, dict_type=True):
        buffer.add(**sars)

    assert len(buffer) == 5

    buffer.clear()
    assert len(buffer) == 0


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


def test_replay_buffer_dump():
    import torch

    # Assign
    filled_buffer = 8
    prop_keys = ["state", "action", "reward", "next_state"]
    buffer = ReplayBuffer(batch_size=5, buffer_size=10)
    for sars in generate_sample_SARS(filled_buffer):
        buffer.add(
            state=torch.tensor(sars[0]),
            reward=sars[1],
            action=[sars[2]],
            next_state=torch.tensor(sars[3]),
            dones=sars[4],
        )

    # Act
    dump = list(buffer.dump_buffer())

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
        sars["state"] = torch.tensor(sars["state"])
        sars["next_state"] = torch.tensor(sars["next_state"])
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
        ser_buffer.append(Experience(**sars))

    # Act
    buffer.load_buffer(ser_buffer)

    # Assert
    samples = buffer.data
    assert len(buffer) == 10
    assert len(samples) == 10
    for sample in samples:
        assert all([hasattr(sample, key) for key in prop_keys])
        assert all([isinstance(getattr(sample, key), list) for key in prop_keys])


def test_replay_buffer_get_state_empty():
    # Assign
    batch_size = 10
    buffer_size = 20
    buffer = ReplayBuffer(batch_size=batch_size, buffer_size=buffer_size)

    # Act
    state: BufferState = buffer.get_state()

    # Assert
    assert state.type == ReplayBuffer.type
    assert state.batch_size == batch_size
    assert state.buffer_size == buffer_size
    assert state.data is None


def test_replay_buffer_get_state_with_data():
    # Assign
    batch_size = 10
    buffer_size = 20
    buffer = ReplayBuffer(batch_size=batch_size, buffer_size=buffer_size)

    for (state, action, reward, next_state, done) in generate_sample_SARS(buffer_size + 1):
        buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

    # Act
    state: BufferState = buffer.get_state()
    state_data: BufferState = buffer.get_state(include_data=True)

    # Assert
    assert state == state_data, "Default option is to include all data"
    assert state.type == ReplayBuffer.type
    assert state.batch_size == batch_size
    assert state.buffer_size == buffer_size
    assert len(state.data) == buffer_size

    for d in state.data:
        b_keys = ("state", "action", "reward", "done", "next_state")
        assert all([k in b_keys for k in d.get_dict().keys()])


def test_replay_buffer_get_state_without_data():
    # Assign
    batch_size = 10
    buffer_size = 20
    buffer = ReplayBuffer(batch_size=batch_size, buffer_size=buffer_size)

    for (state, action, reward, next_state, done) in generate_sample_SARS(buffer_size + 1):
        buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

    # Act
    state: BufferState = buffer.get_state(include_data=False)

    # Assert
    assert state.type == ReplayBuffer.type
    assert state.batch_size == batch_size
    assert state.buffer_size == buffer_size
    assert state.data is None


def test_replay_buffer_from_state_without_data():
    # Assign
    buffer = ReplayBuffer(batch_size=5, buffer_size=20)
    state = buffer.get_state()

    # Act
    new_buffer = ReplayBuffer.from_state(state=state)

    # Assert
    assert new_buffer == buffer
    assert new_buffer.buffer_size == state.buffer_size
    assert new_buffer.batch_size == state.batch_size
    assert new_buffer.data == []


def test_replay_buffer_from_state_with_data():
    # Assign
    buffer = ReplayBuffer(batch_size=5, buffer_size=20)
    buffer = populate_buffer(buffer, 30)
    state = buffer.get_state()

    # Act
    new_buffer = ReplayBuffer.from_state(state=state)

    # Assert
    assert new_buffer == buffer
    assert new_buffer.buffer_size == state.buffer_size
    assert new_buffer.batch_size == state.batch_size
    assert new_buffer.data == state.data
    assert len(buffer.data) == state.buffer_size


def test_replay_buffer_from_state_wrong_type():
    # Assign
    buffer = ReplayBuffer(batch_size=5, buffer_size=20)
    state = buffer.get_state()
    state.type = "WrongType"

    # Act
    with pytest.raises(ValueError):
        ReplayBuffer.from_state(state=state)
