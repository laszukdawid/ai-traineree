# ai-traineree
[![DocStatus](https://readthedocs.org/projects/ai-traineree/badge/?version=latest)](https://ai-traineree.readthedocs.io/)

The intention is to have a zoo of Deep Reinforcment Learning methods and showcasing their application on some environments.
What will distinguish this repo from other DRL is the intention to have types and interfaces, for easier application and understanding, and having all components tested.

## Quick start

To get started with training your RL agent you need three things: an agent, an environment and a runner. Let's say you want to train a DQN agent on OpenAI CartPole-v1:
```
from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.env_runner import EnvRunner
from ai_traineree.tasks import GymTask

task = GymTask('CartPole-v1)
agent = DQNAgent(task.state_size, task.action_size)
env_runner = EnvRunner(task, agent)

scores = env_runner.run()
```
or execute one of provided examples
```
>  python -m examples.cart_dqn
```

That's it.

## Installation

There isn't currently any installation mechanism. Git clone is expected if you want to play yourself. Coming updates include pip package and installation instructions.

As usual with Python, the expectation is to have own virtual environment and then pip install requirements. For example,
```bash
> python -m venv .venv
> git clone git@github.com:laszukdawid/ai-traineree.git
> source .venv/bin/activate
> pip install -r requirements.txt
```

On Ubuntu you might also need to install `swig` (`sudo apt install -y swig`).

## Current state

### Documentation
Even though it's a work in progress, some might find it more useful than going through the code.
As always, feel free to reach out with questions and requests. The best way is through Github issues.

Documentation link: [ReadTheDocs AI-Traineree](https://ai-traineree.readthedocs.io/).

### Playing gym
One way to improve learning speed is to simply show them how to play or, more researchy/creepy, provide a proper seed.
This isn't a general rule, since some algorithms train better without any human interaction, but since you're on GitHub... that's unlikely your case.
Currently there's a script [`interact.py`](scripts/interact.py) which uses OpenAI Gym's play API to record moves and AI Traineree to store them
in a buffer. Such buffers can be loaded by agents on initiation.

This is just a beginning and there will be more work on these interactions.

*Requirement*: Install `pygame`.

### Agents

| Short | Progress | Link | Full name | Doc |
|-------|----------|------|-----------|-----|
| DQN  | [Implemented](ai_traineree/agents/dqn.py) | [DeepMind](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning), [Nature](https://www.nature.com/articles/nature14236)| Deep Q-learning Network  | [Doc](https://ai-traineree.readthedocs.io/en/latest/agents.html#dqn)
| DDPG | [Implemented](ai_traineree/agents/ddpg.py) | [arXiv](https://arxiv.org/abs/1509.02971) | Deep Deterministic Policy Gradient | [Doc](https://ai-traineree.readthedocs.io/en/latest/agents.html#ddpg)
| D4PG | [Implemented](ai_traineree/agents/d4pg.py) | [arXiv](https://arxiv.org/abs/1804.08617) | Distributed Distributional Deterministic Policy Gradients | [Doc](https://ai-traineree.readthedocs.io/en/latest/agents.html#d4pg)
| TD3 | [Implemented](ai_traineree/agents/td3.py) | [arXiv](https://arxiv.org/abs/1802.09477) | Twine Delayed Deep Deterministic policy gradient | [Doc](https://ai-traineree.readthedocs.io/en/latest/agents.html#td3)
| PPO | [Implemented](ai_traineree/agents/ppo.py) | [arXiv](https://arxiv.org/abs/1707.06347) | Proximal Policy Optimization |[Doc](https://ai-traineree.readthedocs.io/en/latest/agents.html#ppo)
| SAC | [Implemented](ai_traineree/agents/sac.py) | [arXiv](https://arxiv.org/abs/1801.01290) | Soft Actor Critic |[Doc](https://ai-traineree.readthedocs.io/en/latest/agents.html#sac)
| TRPO | | [arXiv](https://arxiv.org/abs/1502.05477) | Trust Region Policy Optimization |
| RAINBOW | [Implemented](ai_traineree/agents/rainbow.py) | [arXiv](https://arxiv.org/abs/1710.02298) | DQN with a few improvements |[Doc](https://ai-traineree.readthedocs.io/en/latest/agents.html#rainbow)
| MADDPG | [Implemented](ai_traineree/multi_agents/maddpg.py) | [arXiv](https://arxiv.org/abs/1706.02275) | Multi agent DDPG |[Doc](https://ai-traineree.readthedocs.io/en/latest/multi_agents.html#maddpg)

### Environments

| Name | Progress | Link | Doc |
|------|----------|------|-----|
| OpenAI Gym - Classic | Done |  |
| OpenAI Gym - Atari | Done |  |
| OpenAI Gym - MuJoCo | Not interested. |  |
| Unity ML | Somehow supported. | [Page](https://unity3d.com/machine-learning)
| MAME Linux emulator | Interested. | [Official page](https://www.mamedev.org/)

### Development

| Name | Progress |
|------|----------|
| Gif creation support | Done |
| Tensorboard support | Implemented |
| Pip package | Not started |
| Test coverage > 80% | Tested ~20%, Covered 80% |

There are other potential things on the roadmap but haven't dedicated to them yet. 

Should I focus on something specificallly? Let me know by leaving opening a feature request issue or contacting through [ai-traineree@dawid.lasz.uk](mailto:ai-traineree@dawid.lasz.uk).

