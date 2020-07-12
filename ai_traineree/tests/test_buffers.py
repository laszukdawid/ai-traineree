import numpy as np

from ai_traineree.buffers import ReplayBuffer


def generate_sample_SARS(state_size: int=4, action_size: int=2):
    states = np.random.random((2, state_size))
    action = np.random.random(action_size)
    reward = float(np.random.random() - 0.5)
    done = np.random.random() > 0.5
    return (list(states[0]), list(action), reward, list(states[1]), bool(done))

def test_buffer_size():
    # Assign
    buffer_size = 10
    buffer = ReplayBuffer(batch_size=5, size=buffer_size)

    # Act
    for _ in range(buffer_size+2):
        (state, actions, reward, next_state, done) = generate_sample_SARS()
        buffer.add(state, actions, reward, next_state, done)
    
    # Assert
    assert len(buffer) == buffer_size

def test_buffer_add():
    # Assign
    buffer = ReplayBuffer(batch_size=5, size=5)

    # Act
    assert len(buffer) == 0
    (state, actions, reward, next_state, done) = generate_sample_SARS()
    buffer.add(state, actions, reward, next_state, done)
    
    # Assert
    assert len(buffer) == 1
    
def test_buffer_sample():
    # Assign
    batch_size = 5
    buffer = ReplayBuffer(batch_size=batch_size, size=10)

    # Act
    for _ in range(20):
        (state, actions, reward, next_state, done) = generate_sample_SARS()
        buffer.add(state, actions, reward, next_state, done)
    
    # Assert
    assert len(buffer.sample()) == batch_size