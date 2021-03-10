import numpy as np
import torch
import pytest

from ai_traineree.utils import str_to_list, str_to_number, str_to_seq, str_to_tuple, to_tensor


def test_to_tensor_tensor():
    # Assign
    t = torch.tensor([0, 1, 2, 3])

    # Act
    new_t = to_tensor(t)

    # Assert
    assert torch.equal(new_t, t)


def test_to_tensor_list():
    # Assign
    int_l = [0, 1, 2, 3]
    float_l = [0.5, 1.1, 2.9, 3.0]
    int_l_shape = [list(range(4*i, 4*(i+1))) for i in range(5)]

    # Act
    int_t = to_tensor(int_l)
    float_t = to_tensor(float_l)
    int_t_shape = to_tensor(int_l_shape)

    # Assert
    assert torch.equal(torch.tensor(int_l), int_t)
    assert torch.equal(torch.tensor(float_l), float_t)
    assert torch.equal(torch.tensor(int_l_shape), int_t_shape)

    assert int_t.shape == (4,)
    assert float_t.shape == (4,)
    assert int_t_shape.shape == (5, 4)


def test_to_tensor_list_tesors():
    # Assign
    int_l = [torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 5, 10])]
    float_l = [torch.tensor([0.5, 1.1, 2.9, 3.0]), torch.tensor([0.1, 0.1, 0.1, 10.0])]

    # Act
    int_t = to_tensor(int_l)
    float_t = to_tensor(float_l)

    # Assert
    assert torch.equal(torch.stack(int_l), int_t)
    assert torch.equal(torch.stack(float_l), float_t)

    assert int_t.shape == (2, 4)
    assert float_t.shape == (2, 4)


def test_to_tensor_numpy():
    # Assign
    int_l = np.array([0, 1, 2, 3])
    float_l = np.random.random(4)
    float_l = np.random.random(4)

    # Act
    int_t = to_tensor(int_l)
    float_t = to_tensor(float_l)

    # Assert
    assert torch.equal(torch.tensor(int_l), int_t)
    assert torch.equal(torch.tensor(float_l), float_t)


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
