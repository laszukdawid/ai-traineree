import argparse
import json
import logging
import pprint
import sys

from examples.sagemaker import SageMakerExecutor
from typing import Any, Dict, List

# These hyperparams are set by the SageMaker
PATH_HYPERPARAMETERS = '/opt/ml/input/config/hyperparameters.json'


def read_hyperparameters() -> Dict[str, Any]:
    """Reads SageMaker provided hyperparameters"""
    print("Loading hyperparameters")
    hyperparameters: Dict[str, Any] = {}
    try:
        with open(PATH_HYPERPARAMETERS, 'r') as f:
            hyperparameters.update(json.load(f))
    except Exception:
        logging.error("Loading hyperparameters failed. Returning empty dictionary.")

    print("Loaded hyperparameters:")
    pprint.pprint(hyperparameters)
    return hyperparameters


def parse_arguments(args: List[str]) -> Dict[str, str]:
    print(f"Received args: {args}")
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", help="Name of the environment to solve")
    parser.add_argument("-a", "--agent", help="Name of the agent used to solve the environment (default: PPO)")

    parsed_arguments = parser.parse_args(args)
    return {"example": parsed_arguments.example, "agent": parsed_arguments.agent}


def main():
    hyperparameters = read_hyperparameters()
    config = {}
    if len(sys.argv) > 1:
        config = parse_arguments(args)
    else:
        config = hyperparameters

    env_name = config.get("env", "CartPole-v1")
    agent_name = config.get("agent", "PPO")
    executor = SageMakerExecutor(env_name, agent_name, hyperparameters)
    executor.run()
    executor.save_results("/opt/ml/model/model.pt")


if __name__ == '__main__':
    try:
        print("Provided arguments: " + '\n'.join(sys.argv))
        main()
    # Handle here critical exceptions
    except Exception as e:
        # SageMaker excepts to have failure log here. First 400 chars are return to user on exit(1).
        with open('/opt/ml/output/failure', 'w') as f:
            f.write(f"Training failed with unknown reason\n{str(e)}")
        raise e
