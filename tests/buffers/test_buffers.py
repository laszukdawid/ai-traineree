import numpy

from ai_traineree.buffers import Experience


def test_experience_init():
    # Assign
    state = numpy.random.random(10)
    action = numpy.random.random(2)
    reward = 3
    next_state = numpy.random.random(10)
    done = False

    exp = Experience(state=state, action=action, reward=reward, done=done)
    assert all(exp.state == state)
    assert all(exp.action == action)
    assert exp.reward == reward
    assert exp.done == done
    assert not hasattr(exp, "next_state")

    exp = Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
    assert all(exp.state == state)
    assert all(exp.action == action)
    assert exp.reward == reward
    assert exp.done == done
    assert all(exp.next_state == next_state)


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
