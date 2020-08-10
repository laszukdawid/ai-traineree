# ai-traineree

The intention is to have a zoo of Deep Reinforcment Learning methods and showcasing their application on some environments.
Currently focusing on a couple of popular agents (DQN, DDPG, PPO, A3C) for OpenAi gym environments.

What might distinguish this repo from other DRL is the intention to have types and interfaces, for easier application and understanding, and having all components tested.

## State

Currently implemented agents:
* DQN | Deep Q-learning Network
* DDPG | Deep Deterministic Policy Gradient 
* PPO | Proximal Policy Optimization

Multi agents:
* MADDPG | Multi Agent Deep Deterministic Policy Gradient

... and more to come.

Currently solved environments from OpenAI gyms:
* CartPole-v1
* LunarLander-v2
* LunarLanderContinuous-v2

## Running

Firstly you need to install the package. See below for instructions.

The easiest way to start exploring is through the examples. To run an example, e.g. `cart.py` execute
```bash
> python -m examples.cart
```
from the root directory, i.e. where this README.md.

## Install

As usual with Python, the expectation is to have own virtual environment and then pip install requirements. For example,
```bash
> python -m venv .venv
> source .venv/bin/activate
> pip install -r requirements.txt
```

### Ubuntu

Install debs via `install-ubuntu-debs.sh`, e.g.
```sh
> sudo ./install-ubuntu-debs.sh
```
