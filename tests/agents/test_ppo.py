import copy
import pytest

from ai_traineree.agents.ppo import PPOAgent


def deterministic_interactions(agent, state_size=4, num_iters=50):
    state = [0]*state_size
    next_state = copy.copy(state)
    actions = []
    for i in range(num_iters):
        action = agent.act(state)
        actions.append(action)

        next_state[i % state_size] = (next_state[i % state_size] + 1) % 2
        reward = (i % 4 - 2) / 2.
        done = (i + 1) % 100 == 0

        agent.step(state, action, reward, next_state, done)
        state = copy.copy(next_state)
    return actions


def test_ppo_seed():
    # Assign
    agent = PPOAgent(4, 2, device='cpu')
    agent_clone = copy.deepcopy(agent)

    # Act
    # Make sure agents have the same networks
    assert all([sum(sum(l1.weight - l2.weight)) == 0 for l1, l2 in zip(agent.actor.layers, agent_clone.actor.layers)])
    assert all([sum(sum(l1.weight - l2.weight)) == 0 for l1, l2 in zip(agent.critic.layers, agent_clone.critic.layers)])

    agent.seed(0)
    actions_1 = deterministic_interactions(agent)
    agent_clone.seed(0)
    actions_2 = deterministic_interactions(agent_clone)

    # Assert
    # First we check that there's definitely more than one type of action
    assert actions_1[0] != actions_1[1]
    assert actions_2[0] != actions_2[1]

    # All generated actions need to identical
    for idx, (a1, a2) in enumerate(zip(actions_1, actions_2)):
        assert a1 == pytest.approx(a2, 1e-4), f"Action mismatch on position {idx}: {a1} != {a2}"
