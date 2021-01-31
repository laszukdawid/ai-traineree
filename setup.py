#!/usr/bin/env python
import io
from setuptools import setup

VERSION = "0.1.0"
DESCRIPTION = "Yet another zoo of (Deep) Reinforcment Learning methods in Python using PyTorch"

with io.open('README.md', encoding="utf8") as fp:
    long_description = fp.read().strip()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup_params = dict(
    name="ai-traineree",
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/laszukdawid/ai-traineree",
    author="Dawid Laszuk",
    author_email="ai-traineree@dawid.lasz.uk",
    license="Apache-2.0",
    keywords="AI RL DeRL ML Deep Reinforcment Learning Machine",
    packages=["ai_traineree"],
    python_requires='>3.5,<4',
    install_requires=required,
    extras_require={
        "plot": ["matplotlib"],
        "doc": ["sphinx", "sphinx-rtd-theme"],
        "test": ["mock", "pytest", "pytest-cov", "pytest-mock", "flake8", "gym[box2d]"],
        # Loggers
        "tensorboard": ["tensorboard"],
        "neptune": ["neptune-client", "psutil"],
        # Envs
        # "sneks": ["git+https://github.com/laszukdawid/Sneks.git"],
        "gym": ["gym[all]"],
        "mlagents": ["mlagents"],
        "pettingzoo": ["pettingzoo"],
    },
    test_suite="ai_traineree.tests",
)

dist = setup(**setup_params)
