import random
import warnings

import mock
import numpy as np
import pytest
import torch

from aitraineree.types.dataspace import DataSpace
from tests.utils import deterministic_interactions, fake_step, feed_agent

__all__ = ["deterministic_interactions", "fake_step", "feed_agent"]


def _is_cuda_runtime_supported() -> bool:
    if not torch.cuda.is_available():
        return False

    try:
        device = torch.device("cuda:0")
        probe = torch.randn((2, 2), device=device)
        _ = probe @ probe
        torch.cuda.synchronize(device)
        return True
    except Exception as exc:
        warnings.warn(
            f"CUDA runtime is unavailable or unsupported ({exc}). Tests marked 'requires_cuda' will be skipped.",
            RuntimeWarning,
        )
        return False


CUDA_RUNTIME_SUPPORTED = _is_cuda_runtime_supported()


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda: test requires a working CUDA runtime")


def pytest_collection_modifyitems(config, items):
    if CUDA_RUNTIME_SUPPORTED:
        return

    skip_cuda = pytest.mark.skip(reason="CUDA runtime unavailable or unsupported on this machine")
    for item in items:
        if "requires_cuda" in item.keywords:
            item.add_marker(skip_cuda)


class MockContinuousSpace:
    def __init__(self, *args):
        self.shape = args


class MockDiscreteSpace:
    def __init__(self, n):
        self.n = n

    def __call__(self):
        return self.n


def rnd_state():
    return random.choices(range(10), k=5)


@pytest.fixture
def fix_env_discrete():
    mock_env = mock.Mock()
    mock_env.reset.return_value = (rnd_state(), {})
    mock_env.step.return_value = (rnd_state(), 0, False, False, "")
    mock_env.observation_space = np.array((4, 2))
    mock_env.action_space = MockDiscreteSpace(2)
    return mock_env


@pytest.fixture
def fix_env():
    mock_env = mock.Mock()
    mock_env.reset.return_value = (rnd_state(), {})
    mock_env.step.return_value = (rnd_state(), 0, False, False, "")
    mock_env.observation_space = np.array((4, 2))
    mock_env.action_space = MockContinuousSpace(2, 4)
    return mock_env


@pytest.fixture
def float_1d_space():
    return DataSpace(dtype="float", shape=(5,), low=-2, high=2)


@pytest.fixture
def int_1d_space():
    return DataSpace(dtype="int", shape=(1,), low=0, high=4)
