from pprint import pprint
from typing import Any

import torch

from ai_traineree.agents.dqn import DQNAgent as Agent
from ai_traineree.loggers.tensorboard_logger import TensorboardLogger
from ai_traineree.runners.env_runner import EnvRunner
from ai_traineree.tasks import GymTask

config_default = {"hidden_layers": (50, 50)}
config_updates = [{"n_steps": n} for n in range(1, 11)]

task = GymTask("CartPole-v1")
seeds = [32167, 1, 999, 2833700, 13]

for idx, config_update in enumerate(config_updates):
    config: dict[str, Any] = config_default.copy()
    config.update(config_update)

    for seed in seeds:
        config["seed"] = seed
        pprint(config)
        torch.manual_seed(config["seed"])
        agent = Agent(task.obs_size, task.action_size, **config)

        data_logger = TensorboardLogger(log_dir=f"runs/MultiExp-{task.name}-i{idx}-s{seed}")
        env_runner = EnvRunner(task, agent, data_logger=data_logger)
        env_runner.seed(seed)
        env_runner.run(reward_goal=99999, max_episodes=500, eps_decay=0.95, force_new=True)
        data_logger.close()
