import random
import string
from typing import Any

from ai_traineree.types.experience import Experience


def r_string(n: int) -> str:
    return "".join(random.choices(string.printable, k=n))


def r_float(n: int) -> list[float]:
    return [random.random() for _ in range(n)]


def test_experience_init_required():
    # Assign
    required_data = {k: random.random() for k in ["obs", "action", "reward", "done", "next_obs"]}

    # Act
    exp = Experience(**required_data)

    # Assert
    assert exp.data == required_data


def test_experience_init_extra():
    # Assign
    all_data = {k: random.random() for k in Experience.whitelist}

    # Act
    exp = Experience(**all_data)

    # Assert
    assert exp.data == all_data


def test_experience_init_not_whitelisted():
    data = {r_string(10): r_float(5) for _ in range(10)}

    exp = Experience(**data)

    for key in data.keys():
        assert key not in exp.data


def test_experience_init():
    # Assign
    state = r_float(10)
    action = r_float(2)
    reward = 3
    next_state = r_float(10)
    done = False

    exp = Experience(state=state, action=action, reward=reward, done=done)
    assert exp.state == state
    assert exp.action == action
    assert exp.reward == reward
    assert exp.done == done
    assert not hasattr(exp, "next_state")

    exp = Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
    assert exp.state == state
    assert exp.action == action
    assert exp.reward == reward
    assert exp.done == done
    assert exp.next_state == next_state


def test_experience_comparison():
    # Assign
    e1a = Experience(state=[0, 2], action=2, done=False)
    e1b = Experience(state=[0, 2], action=2, done=False)
    e1c = Experience(action=2, done=False, state=[0, 2])
    e2 = Experience(state=[0, 2, 3], action=[1, 1], done=False)
    e3 = Experience(state=[0, 2], action=2, done=True)

    # Assert
    assert e1a == e1b
    assert e1b == e1c
    assert e2 != e3


def test_experience_update():
    # Assign
    exp = Experience(obs=1)

    # Act
    exp.update(action=42, next_obs=123)

    # Assert
    assert exp.obs == 1
    assert exp.action == 42
    assert exp.next_obs == 123


def test_experience_update_not_whitelisted():
    exp = Experience(state=123, action=456, weight=0)
    data = {r_string(10): r_float(5) for _ in range(10)}

    # Act
    exp.update(**data)

    # Assert
    for key in data.keys():
        assert key not in exp.data


def test_experience_get_dict():
    import numpy
    import torch

    # Assign
    init_data: dict[str, Any] = {k: r_float(5) for k in ["obs", "action", "reward", "done", "next_obs"]}
    init_data["state"] = numpy.random.random(10)
    init_data["advantage"] = torch.rand(10)
    exp = Experience(**init_data)

    # Act
    out_data = exp.get_dict()

    # Assert
    assert out_data == init_data, "Both are dictionaries"


def test_experience_get_dict_serialize():
    import numpy
    import torch

    # Assign
    init_data: dict[str, Any] = {k: r_float(5) for k in ["action", "reward", "done"]}
    init_data["obs"] = numpy.random.random(5)
    init_data["next_obs"] = torch.rand(10)
    exp = Experience(**init_data)

    # Act
    out_data = exp.get_dict(serialize=True)

    # Assert
    assert all([init_data[k] == out_data[k] for k in ["action", "reward", "done"]])
    assert init_data["obs"].tolist() == out_data["obs"]
    assert init_data["next_obs"].tolist() == out_data["next_obs"]
