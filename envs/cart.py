import time

def interact_env(env, agent, eps, render=False, max_t=1000) -> int:
    score = 0
    state = env.reset()
    for _ in range(max_t):
        if render:
            env.render()
            time.sleep(0.05)
        action = agent.act(state, eps)
        next_state, reward, done, _ = env.step(action)
        score += reward
        agent.step(state, action, score, next_state, done)
        state = next_state
        if done:
            # print(f"\niter_t: {iter_t}")
            break
    return score
