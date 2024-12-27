import json
import random

import numpy as np
import pytest
import torch

from ai_traineree.types.dataspace import DataSpace
from ai_traineree.utils import (
    condens_ndarray,
    serialize,
    str_to_list,
    str_to_number,
    str_to_seq,
    str_to_tuple,
    to_numbers_seq,
    to_tensor,
)
from conftest import deterministic_interactions


def generate_value():
    if random.random() < 0.5:
        return [random.random() for _ in range(random.randint(0, 10))]
    return random.random()


def test_to_tensor_tensor():
    # Assign
    t = torch.tensor([0, 1, 2, 3])

    # Act
    new_t = to_tensor(t)

    # Assert
    assert torch.equal(new_t, t)


@pytest.mark.parametrize(
    "test_input,expected_shape",
    [
        ([0, 1, 2, 3], (4,)),
        ([0.5, 1.1, 2.9, 3.0], (4,)),
        ([list(range(4 * i, 4 * (i + 1))) for i in range(5)], (5, 4)),
    ],
)
def test_to_tensor_list(test_input, expected_shape):
    # Act
    out = to_tensor(test_input)

    # Assert
    assert torch.equal(out, torch.tensor(test_input))
    assert out.shape == expected_shape


@pytest.mark.parametrize(
    "test_input,expected_shape",
    [
        ([torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 5, 10])], (2, 4)),
        ([torch.tensor([0.5, 1.1, 2.9, 3.0]), torch.tensor([0.1, 0.1, 0.1, 10.0])], (2, 4)),
    ],
)
def test_to_tensor_list_tesors(test_input, expected_shape):
    # Act
    out = to_tensor(test_input)

    # Assert
    assert torch.equal(torch.stack(test_input), out)
    assert out.shape == expected_shape


@pytest.mark.parametrize(
    "test_input",
    [
        np.array([0, 1, 2, 3]),  # int
        np.random.random(4).astype(dtype=np.float16),  # float
        np.random.random(4).astype(dtype=np.float32),  # float
    ],
)
def test_to_tensor_numpy(test_input):
    # Act
    out = to_tensor(test_input)

    # Assert
    assert torch.equal(torch.tensor(test_input), out)


@pytest.mark.parametrize(
    "test_input",
    [
        [np.array([0, 1, 2, 3]), np.array([1, 2, 5, 10])],
        [np.random.random(4), np.random.random(4)],
    ],
)
def test_to_tensor_list_numpy(test_input):
    # Act
    out = to_tensor(test_input)

    # Assert
    assert isinstance(out, torch.Tensor)
    assert out.shape == (len(test_input), len(test_input[0]))


def test_str_to_number():
    assert 1 == str_to_number("1")
    assert 1 == str_to_number(" 1 ")
    assert 1.5 == str_to_number(" 1.5 ")

    with pytest.raises(ValueError):
        str_to_list("[0, 1")


def test_str_to_list():
    assert [0.1, 0, 0, 2.5, 1] == str_to_list("[0.1, 0, 0, 2.5, 1]")
    assert [0.1, 1] == str_to_list("[0.1, 1]")
    assert [0, 1] == str_to_list("[0,1]")

    with pytest.raises(ValueError):
        str_to_list("[0, 1")
    with pytest.raises(ValueError):
        str_to_list("0, 1]")
    with pytest.raises(ValueError):
        str_to_list("[0, 1)")


def test_str_to_tuple():
    assert (1, 4) == str_to_tuple("(1, 4)")
    assert (1, 4) == str_to_tuple("1, 4")
    assert (1, 4) == str_to_tuple("1,4      ")
    assert (4,) == str_to_tuple("4")

    with pytest.raises(ValueError):
        str_to_tuple("(0, 1")
    with pytest.raises(ValueError):
        str_to_tuple("0, 1)")
    with pytest.raises(ValueError):
        str_to_tuple("[0, 1)")


def test_str_to_seq():
    assert [0.1, 0] == str_to_seq("[0.1, 0]") == str_to_list("[0.1, 0]")
    assert [0, 1] == str_to_list("[0,1]") == str_to_list("[0,1]")
    assert (1, 4) == str_to_seq("(1, 4)") == str_to_tuple("(1, 4)")
    assert (1, 4) == str_to_seq("1, 4") == str_to_tuple("1, 4")
    assert (1, 4) == str_to_seq("1,4        ") == str_to_tuple("1,4      ")
    assert (4,) == str_to_seq("4")

    with pytest.raises(ValueError):
        str_to_seq("(0, 1")
    with pytest.raises(ValueError):
        str_to_seq("0, 1)")
    with pytest.raises(ValueError):
        str_to_seq("[0, 1)")
    with pytest.raises(ValueError):
        str_to_seq("{0, 1}")


def test_serialize_experience_individual():
    from ai_traineree.types.experience import Experience

    for key in Experience.whitelist:
        value = generate_value()
        e = Experience(**{key: value})

        ser = serialize(e)
        assert ser == '{"%s": "%s"}' % (key, value)


def test_serialize_experience_all_keys():
    import json

    from ai_traineree.types.experience import Experience

    state = {key: generate_value() for key in Experience.whitelist}
    e = Experience(**state)

    # Act
    ser = serialize(e)

    # Assert
    assert json.loads(ser) == {k: str(v) for (k, v) in state.items()}


def test_serialize_buffer_state_list():
    from ai_traineree.buffers import BufferState
    from ai_traineree.types.experience import Experience

    def generate_exp() -> Experience:
        return Experience(
            state=np.random.random(10).tolist(),
            next_state=np.random.random(10).tolist(),
            action=np.random.random(3).tolist(),
            reward=float(random.random()),
            done=bool(random.randint(0, 8) == 8),
        )

    # Assign
    buffer_size = 1e1
    buffer_state = BufferState(type="Type", batch_size=200, buffer_size=buffer_size)
    buffer_state.data = [generate_exp() for _ in range(int(buffer_size))]

    # Act
    ser = serialize(buffer_state)

    # Assert
    des = json.loads(ser)
    assert set(des.keys()) == set(("type", "buffer_size", "batch_size", "data", "extra"))
    assert len(des["data"]) == buffer_size
    assert des["type"] == "Type"
    assert des["extra"] is None


def test_serialize_buffer_state_numpy():
    from ai_traineree.buffers import BufferState
    from ai_traineree.types.experience import Experience

    def generate_exp() -> Experience:
        return Experience(
            state=np.random.random(10),
            next_state=np.random.random(10),
            action=np.random.random(3),
            reward=random.random(),
            done=(random.randint(0, 8) == 8),
        )

    # Assign
    buffer_size = int(1e1)
    buffer_state = BufferState(type="Type", batch_size=200, buffer_size=buffer_size)
    buffer_state.data = [generate_exp() for _ in range(int(buffer_size))]

    # Act
    ser = serialize(buffer_state)

    # Assert
    des = json.loads(ser)
    assert set(des.keys()) == set(("type", "buffer_size", "batch_size", "data", "extra"))
    assert len(des["data"]) == buffer_size
    assert des["type"] == "Type"
    assert des["extra"] is None


def test_serialize_network_state_actual():
    from ai_traineree.agents.dqn import DQNAgent

    agent = DQNAgent(DataSpace(dtype="float", shape=(4,)), DataSpace("int", (1,), low=0, high=2))
    deterministic_interactions(agent, 30)
    network_state = agent.get_network_state()

    # Act
    ser = serialize(network_state)

    # Assert
    des = json.loads(ser)
    assert set(des["net"].keys()) == set(("target_net", "net"))


def test_serialize_agent_state_actual():
    from ai_traineree.agents.dqn import DQNAgent

    agent = DQNAgent(DataSpace(dtype="float", shape=(4,)), DataSpace("int", (1,), low=0, high=2))
    deterministic_interactions(agent, 30)
    state = agent.get_state()

    # Act
    ser = serialize(state)

    # Assert
    des = json.loads(ser)
    assert des["model"] == DQNAgent.model
    assert len(des["buffer"]["data"]) == 30
    assert set(des["network"]["net"].keys()) == set(("target_net", "net"))


def test_to_numbers_seq_tuple_list():
    t_in = (1, 2, 3)
    l_in = [1, 2, 3]

    t_out = to_numbers_seq(t_in)
    l_out = to_numbers_seq(l_in)

    assert t_out == t_in
    assert l_out == l_in


def test_to_numbers_seq_str():
    s_t1_in = "(1,2,3)"
    s_t2_in = "(1, 2, 3)"
    s_l1_in = "[1,2,3]"
    s_l2_in = "[1, 2, 3]"

    s_t1_out = to_numbers_seq(s_t1_in)
    s_t2_out = to_numbers_seq(s_t2_in)
    s_l1_out = to_numbers_seq(s_l1_in)
    s_l2_out = to_numbers_seq(s_l2_in)

    assert s_t1_out == (1, 2, 3)
    assert s_t2_out == (1, 2, 3)
    assert s_l1_out == [1, 2, 3]
    assert s_l2_out == [1, 2, 3]


def test_to_numbers_seq_num():
    int_in = 2
    float_in = 2.123

    int_out = to_numbers_seq(int_in)
    float_out = to_numbers_seq(float_in)

    assert int_out == (2,)
    assert float_out == (2.123,)


def test_to_numbers_seq_unknown():
    test_cases = ["asd", "[2,1", {"key": "value"}]

    for test_case in test_cases:
        with pytest.raises(ValueError):
            to_numbers_seq(test_case)


def test_condens_ndarray_possible():
    l_shape = []
    for shape_idx in range(5, 1):
        l_shape.append(shape_idx)
        shape = tuple(l_shape)
        a = np.zeros(shape)
        assert 0 == condens_ndarray(a)

        a = np.ones(shape) * 1.5
        assert 0.5 == condens_ndarray(a)


def test_condens_ndarray_not_possible():
    a = np.arange(10)
    assert np.all(a.copy() == condens_ndarray(a.copy()))

    a = np.random.random((3, 4, 2))
    assert np.all(a.copy() == condens_ndarray(a.copy()))


if __name__ == "__main__":
    test_serialize_network_state_actual()
