"""
Tests for Agent modules in section_1/agent/
"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'project_2_code', 'section_1'))

from project_2_code.section_1.utils.config import Config
from project_2_code.section_1.component.envs import Task
from project_2_code.section_1.component.network import (
    FCBody,
    CategoricalPolicyNet,
    CategoricalActorCriticNet,
    GaussianActorCriticNet
)
from project_2_code.section_1.agent.REINFORCE_agent import REINFORCEAgent
from project_2_code.section_1.agent.A2C_agent import A2CAgent
from project_2_code.section_1.utils.torch_utils import get_time_str


def _make_cartpole_config():
    """Create a minimal CartPole config for testing."""
    config = Config()
    config.game = 'CartPole-v1'
    config.seed = 42
    config.time = get_time_str()
    config.log_level = 0
    config.tag = 'test'

    config.task_fn = lambda: Task(config.game, config.seed)
    config.eval_env = Task(config.game, config.seed)

    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.001)

    config.discount = 0.99
    config.episode_length = 200
    config.eval_episodes = 1
    config.max_steps = 10  # very small for testing

    return config


def _make_reinforce_cartpole_config():
    """Create CartPole config for REINFORCE agent."""
    config = _make_cartpole_config()
    config.network_fn = lambda: CategoricalPolicyNet(
        config.action_dim, actor_body=FCBody(config.state_dim, hidden_units=(64, 64)))
    return config


def _make_a2c_cartpole_config():
    """Create CartPole config for A2C agent."""
    config = _make_cartpole_config()
    config.network_fn = lambda: CategoricalActorCriticNet(
        config.action_dim,
        actor_body=FCBody(config.state_dim),
        critic_body=FCBody(config.state_dim))
    config.entropy_weight = 0.01
    config.value_loss_weight = 1
    config.gradient_clip = 5
    config.rollout_length = 10  # small for testing
    return config


class TestREINFORCEAgent:
    """Tests for REINFORCEAgent."""

    def test_reinforce_init(self, set_cpu_device):
        """Test REINFORCEAgent initialization."""
        config = _make_reinforce_cartpole_config()
        agent = REINFORCEAgent(config)

        assert agent.total_steps == 0
        assert agent.network is not None
        assert agent.optimizer is not None
        assert agent.task is not None
        agent.close()

    def test_reinforce_step(self, set_cpu_device):
        """Test REINFORCEAgent can perform one step without error."""
        config = _make_reinforce_cartpole_config()
        agent = REINFORCEAgent(config)

        # Perform one step - should not raise any error
        agent.step()

        # After one step, total_steps should have increased
        assert agent.total_steps > 0
        agent.close()

    def test_reinforce_multiple_steps(self, set_cpu_device):
        """Test REINFORCEAgent can perform multiple steps."""
        config = _make_reinforce_cartpole_config()
        agent = REINFORCEAgent(config)

        for _ in range(3):
            agent.step()

        assert agent.total_steps > 0
        agent.close()


class TestA2CAgent:
    """Tests for A2CAgent."""

    def test_a2c_init(self, set_cpu_device):
        """Test A2CAgent initialization."""
        config = _make_a2c_cartpole_config()
        agent = A2CAgent(config)

        assert agent.total_steps == 0
        assert agent.network is not None
        assert agent.optimizer is not None
        agent.close()

    def test_a2c_step(self, set_cpu_device):
        """Test A2CAgent can perform one step without error."""
        config = _make_a2c_cartpole_config()
        agent = A2CAgent(config)

        # Perform one step - should not raise any error
        agent.step()

        # After one step, total_steps should have increased
        assert agent.total_steps > 0
        agent.close()

    def test_a2c_multiple_steps(self, set_cpu_device):
        """Test A2CAgent can perform multiple steps."""
        config = _make_a2c_cartpole_config()
        agent = A2CAgent(config)

        for _ in range(3):
            agent.step()

        assert agent.total_steps > 0
        agent.close()


class TestBaseAgentEval:
    """Tests for BaseAgent.eval_episode()."""

    def test_eval_episode_returns_number(self, set_cpu_device):
        """Test eval_episode returns a numeric reward."""
        config = _make_reinforce_cartpole_config()
        agent = REINFORCEAgent(config)

        ret = agent.eval_episode()
        assert isinstance(ret, (int, float))
        agent.close()

    def test_eval_episode_finite(self, set_cpu_device):
        """Test eval_episode returns a finite reward."""
        config = _make_reinforce_cartpole_config()
        agent = REINFORCEAgent(config)

        ret = agent.eval_episode()
        import math
        assert math.isfinite(ret)
        agent.close()