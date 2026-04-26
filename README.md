# ai-traineree
[![DocStatus](https://readthedocs.org/projects/ai-traineree/badge/?version=latest)](https://ai-traineree.readthedocs.io/)
[![codecov](https://codecov.io/gh/laszukdawid/ai-traineree/branch/master/graph/badge.svg?token=S62DK7HPYA)](https://codecov.io/gh/laszukdawid/ai-traineree)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/00563b8422454e10bb4ffab64068aa62)](https://www.codacy.com/gh/laszukdawid/ai-traineree/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=laszukdawid/ai-traineree&amp;utm_campaign=Badge_Grade)
[![Conda](https://anaconda.org/conda-forge/ai-traineree/badges/version.svg)](https://anaconda.org/conda-forge/ai-traineree/badges/version.svg)

The intention is to have a zoo of Deep Reinforcment Learning methods and showcasing their application on some environments.

Read more in the doc: [ReadTheDocs AI-Traineree](https://ai-traineree.readthedocs.io/).

![CartPole-v1](./static/CartPole-v1.gif)
![Snek](./static/hungrysnek.gif)

## Why another Deep Reinforcement Learning framework?

The main reason is the implemention philosophy.
We strongly believe that agents should be emerged in the environment and not the other way round.
Majority of the popular implementations pass environment instance to the agent as if the agent was the focus point.
This might ease implementation of some algorithms but it isn't representative of the world;
agents want to control the environment but that doesn't mean they can/should.

That, and using PyTorch instead of Tensorflow or JAX.

## Quick start

To get started with training your RL agent you need three things: an agent, an environment and a runner. Let's say you want to train a DQN agent on Gymnasium CartPole-v1:

```python
from aitraineree.agents.dqn import DQNAgent
from aitraineree.runners.env_runner import EnvRunner
from aitraineree.tasks import GymTask

task = GymTask('CartPole-v1')
agent = DQNAgent(task.obs_space, task.action_space)
env_runner = EnvRunner(task, agent)

scores = env_runner.run()
```

or execute one of provided examples

```sh
python -m examples.cart_dqn
```

That's it.

## Installation

### PyPi (recommended)

The quickest way to install package is through `pip`.

```sh
pip install ai-traineree
```

In case you're using [uv](https://docs.astral.sh/uv/) which is recommended as it makes building environments much faster, use

```sh
uv add ai-traineree
# or
# uv pip install ai-traineree
```

### Conda

AI Traineree is also available in Conda via conda-forge channel

```sh
conda install -c conda-forge ai-traineree
```

Source: https://anaconda.org/conda-forge/ai-traineree

### Git repository clone

As usual with Python, the expectation is to have own virtual environment and then install project dependencies. For example,

```bash
git clone git@github.com:laszukdawid/ai-traineree.git
cd ai-traineree
python -m venv .venv
source .venv/bin/activate
pip install -e .
# or, with uv
# uv sync
```

## Current state

### Playing gym

One way to improve learning speed is to simply show them how to play or, more researchy/creepy, provide a proper seed.
This isn't a general rule, since some algorithms train better without any human interaction, but since you're on GitHub... that's unlikely your case.
Currently there's a script [`interact.py`](scripts/interact.py) which uses Gymnasium's play API to record moves and AI Traineree to store them
in a buffer. Such buffers can be loaded by agents on initiation.

This is just a beginning and there will be more work on these interactions.

*Requirement*: Install `pygame`.

### Agents

| Short   | Progress                                      | Link                                                                                                           | Full name                                                 | Doc                                                                      |
| ------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------ |
| DQN     | [Implemented](aitraineree/agents/dqn.py)     | [DeepMind](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning) | Deep Q-learning Network                                   | [Doc](https://ai-traineree.readthedocs.io/en/latest/agents.html#dqn)     |
| DDPG    | [Implemented](aitraineree/agents/ddpg.py)    | [arXiv](https://arxiv.org/abs/1509.02971)                                                                      | Deep Deterministic Policy Gradient                        | [Doc](https://ai-traineree.readthedocs.io/en/latest/agents.html#ddpg)    |
| D4PG    | [Implemented](aitraineree/agents/d4pg.py)    | [arXiv](https://arxiv.org/abs/1804.08617)                                                                      | Distributed Distributional Deterministic Policy Gradients | [Doc](https://ai-traineree.readthedocs.io/en/latest/agents.html#d4pg)    |
| TD3     | [Implemented](aitraineree/agents/td3.py)     | [arXiv](https://arxiv.org/abs/1802.09477)                                                                      | Twine Delayed Deep Deterministic policy gradient          | [Doc](https://ai-traineree.readthedocs.io/en/latest/agents.html#td3)     |
| PPO     | [Implemented](aitraineree/agents/ppo.py)     | [arXiv](https://arxiv.org/abs/1707.06347)                                                                      | Proximal Policy Optimization                              | [Doc](https://ai-traineree.readthedocs.io/en/latest/agents.html#ppo)     |
| SAC     | [Implemented](aitraineree/agents/sac.py)     | [arXiv](https://arxiv.org/abs/1801.01290)                                                                      | Soft Actor Critic                                         | [Doc](https://ai-traineree.readthedocs.io/en/latest/agents.html#sac)     |
| TRPO    |                                               | [arXiv](https://arxiv.org/abs/1502.05477)                                                                      | Trust Region Policy Optimization                          |
| RAINBOW | [Implemented](aitraineree/agents/rainbow.py) | [arXiv](https://arxiv.org/abs/1710.02298)                                                                      | DQN with a few improvements                               | [Doc](https://ai-traineree.readthedocs.io/en/latest/agents.html#rainbow) |

### Multi agents

We provide both Multi Agents agents entities and means to execute them against supported (below) environements.
However, that doesn't mean one can be used without the other.

| Short  | Progress                                          | Link                                      | Full name              | Doc                                                                          |
| ------ | ------------------------------------------------- | ----------------------------------------- | ---------------------- | ---------------------------------------------------------------------------- |
| IQL    | [Implemented](aitraineree/multi_agents/iql.py)    |                                           | Independent Q-Learners | [Doc](https://ai-traineree.readthedocs.io/en/latest/multi_agent.html#iql)    |
| MADDPG | [Implemented](aitraineree/multi_agents/maddpg.py) | [arXiv](https://arxiv.org/abs/1706.02275) | Multi agent DDPG       | [Doc](https://ai-traineree.readthedocs.io/en/latest/multi_agent.html#maddpg) |

### Loggers

Supports using Tensorboard (via PyTorch's [SummaryWriter](https://pytorch.org/docs/stable/tensorboard.html)) to display metrics. A lightweight `FileLogger` is also available for local experiment data.

### Environments

| Name                    | Progress           | Link                                                                                         |
| ----------------------- | ------------------ | -------------------------------------------------------------------------------------------- |
| Gymnasium - Classic     | Done               |
| Gymnasium - Atari       | Done               |
| Gymnasium - MuJoCo      | Not interested.    |
| PettingZoo              | Initial support    | [Page](https://www.pettingzoo.ml/) / [GitHub](https://github.com/PettingZoo-Team/PettingZoo) |
| Unity ML                | Somehow supported. | [Page](https://unity3d.com/machine-learning)                                                 |
| MAME Linux emulator     | Interested.        | [Official page](https://www.mamedev.org/)                                                    |

### Development

We are open to any contributions. If you want to contribute but don't know what then feel free to reach out (see Contact below).
The best way to start is through updating documentation and adding tutorials.
In addition there are many other things that we know of which need improvement but also plenty that we don't know of.

Setting up development environment is easiest with [uv](https://docs.astral.sh/uv/), using the `dev` dependency group:

```bash
uv sync --dev
```

Typical development commands:

```bash
uvx ruff@0.3.0 check
uv run pytest
```

### Contact

Should we focus on something specificallly? Let us know by opening a feature request [GitHub issue](https://github.com/laszukdawid/ai-traineree/issues) or contacting through [ai-traineree@dawid.lasz.uk](mailto:ai-traineree@dawid.lasz.uk).

## Citing project

```latex
@misc{ai-traineree,
  author = {Laszuk, Dawid},
  title = {AI Traineree: Reinforcement learning toolset},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/laszukdawid/ai-traineree}},
}
```
