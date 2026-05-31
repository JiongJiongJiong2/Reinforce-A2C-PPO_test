"""
Pytest configuration and fixtures for RL project tests.
"""
import sys
import os
import pytest

# Add project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import gymnasium as gym


@pytest.fixture(autouse=True)
def create_log_dirs():
    """Create log directories before tests."""
    os.makedirs('./log', exist_ok=True)
    os.makedirs('./tf_log', exist_ok=True)


@pytest.fixture
def set_cpu_device():
    """Force CPU device for testing."""
    from project_2_code.section_1.utils.config import Config
    Config.DEVICE = torch.device('cpu')
    return Config.DEVICE


@pytest.fixture
def cartpole_env():
    """Create CartPole environment for testing."""
    env = gym.make('CartPole-v1')
    yield env
    env.close()


@pytest.fixture
def pendulum_env():
    """Create Pendulum environment for testing."""
    env = gym.make('Pendulum-v1')
    yield env
    env.close()