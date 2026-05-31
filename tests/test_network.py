"""
Tests for network modules in section_1/component/network.py
"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'project_2_code', 'section_1'))

from project_2_code.section_1.component.network import (
    FCBody,
    GaussianPolicyNet,
    CategoricalPolicyNet,
    GaussianActorCriticNet,
    CategoricalActorCriticNet
)
from project_2_code.section_1.utils.config import Config


class TestFCBody:
    """Tests for FCBody network."""

    def test_fcbody_forward_shape(self, set_cpu_device):
        """Test FCBody output shape matches feature_dim."""
        state_dim = 4
        hidden_units = (64, 64)
        body = FCBody(state_dim, hidden_units)

        batch_size = 8
        x = torch.randn(batch_size, state_dim)
        output = body(x)

        assert output.shape == (batch_size, hidden_units[-1])
        assert body.feature_dim == hidden_units[-1]

    def test_fcbody_single_input(self, set_cpu_device):
        """Test FCBody with single input."""
        state_dim = 4
        body = FCBody(state_dim, (64, 64))

        x = torch.randn(1, state_dim)
        output = body(x)

        assert output.shape == (1, 64)


class TestGaussianPolicyNet:
    """Tests for GaussianPolicyNet (continuous action space)."""

    def test_gaussian_policy_forward_shape(self, set_cpu_device):
        """Test GaussianPolicyNet output shapes."""
        state_dim = 4
        action_dim = 2
        hidden_units = (64, 64)

        actor_body = FCBody(state_dim, hidden_units)
        policy = GaussianPolicyNet(action_dim, actor_body)

        batch_size = 4
        obs = torch.randn(batch_size, state_dim)
        output = policy(obs)

        # Check all required keys exist
        assert 'action' in output
        assert 'log_pi_a' in output
        assert 'entropy' in output
        assert 'mean' in output

        # Check shapes
        assert output['action'].shape == (batch_size, action_dim)
        assert output['log_pi_a'].shape == (batch_size, 1)
        assert output['entropy'].shape == (batch_size, 1)
        assert output['mean'].shape == (batch_size, action_dim)

    def test_gaussian_policy_mean_range(self, set_cpu_device):
        """Test GaussianPolicyNet mean is in valid range (tanh output)."""
        state_dim = 4
        action_dim = 2

        actor_body = FCBody(state_dim, (64, 64))
        policy = GaussianPolicyNet(action_dim, actor_body)

        obs = torch.randn(10, state_dim)
        output = policy(obs)

        # Mean should be in [-1, 1] due to tanh
        assert output['mean'].min() >= -1.0
        assert output['mean'].max() <= 1.0

    def test_gaussian_policy_log_prob_finite(self, set_cpu_device):
        """Test GaussianPolicyNet log probabilities are finite."""
        state_dim = 4
        action_dim = 2

        actor_body = FCBody(state_dim, (64, 64))
        policy = GaussianPolicyNet(action_dim, actor_body)

        obs = torch.randn(10, state_dim)
        output = policy(obs)

        assert torch.isfinite(output['log_pi_a']).all()
        assert torch.isfinite(output['entropy']).all()


class TestCategoricalPolicyNet:
    """Tests for CategoricalPolicyNet (discrete action space)."""

    def test_categorical_policy_forward_shape(self, set_cpu_device):
        """Test CategoricalPolicyNet output shapes."""
        state_dim = 4
        action_dim = 3  # e.g., CartPole has 2 actions

        actor_body = FCBody(state_dim, (64, 64))
        policy = CategoricalPolicyNet(action_dim, actor_body)

        batch_size = 8
        obs = torch.randn(batch_size, state_dim)
        output = policy(obs)

        # Check all required keys exist
        assert 'action' in output
        assert 'log_pi_a' in output
        assert 'entropy' in output

        # Check shapes
        assert output['action'].shape == (batch_size,)
        assert output['log_pi_a'].shape == (batch_size, 1)
        assert output['entropy'].shape == (batch_size, 1)

    def test_categorical_policy_action_valid(self, set_cpu_device):
        """Test CategoricalPolicyNet actions are valid indices."""
        state_dim = 4
        action_dim = 3

        actor_body = FCBody(state_dim, (64, 64))
        policy = CategoricalPolicyNet(action_dim, actor_body)

        obs = torch.randn(20, state_dim)
        output = policy(obs)

        # Actions should be in [0, action_dim)
        assert (output['action'] >= 0).all()
        assert (output['action'] < action_dim).all()


class TestGaussianActorCriticNet:
    """Tests for GaussianActorCriticNet (continuous action space with value)."""

    def test_gaussian_actor_critic_forward_shape(self, set_cpu_device):
        """Test GaussianActorCriticNet output shapes."""
        state_dim = 4
        action_dim = 2

        actor_body = FCBody(state_dim, (64, 64))
        critic_body = FCBody(state_dim, (64, 64))
        network = GaussianActorCriticNet(action_dim, actor_body, critic_body)

        batch_size = 6
        obs = torch.randn(batch_size, state_dim)
        output = network(obs)

        # Check all required keys exist
        assert 'action' in output
        assert 'log_pi_a' in output
        assert 'entropy' in output
        assert 'mean' in output
        assert 'v' in output

        # Check shapes
        assert output['action'].shape == (batch_size, action_dim)
        assert output['log_pi_a'].shape == (batch_size, 1)
        assert output['entropy'].shape == (batch_size, 1)
        assert output['mean'].shape == (batch_size, action_dim)
        assert output['v'].shape == (batch_size, 1)

    def test_gaussian_actor_critic_params(self, set_cpu_device):
        """Test GaussianActorCriticNet has separate actor and critic params."""
        state_dim = 4
        action_dim = 2

        actor_body = FCBody(state_dim, (64, 64))
        critic_body = FCBody(state_dim, (64, 64))
        network = GaussianActorCriticNet(action_dim, actor_body, critic_body)

        # Check that actor_params and critic_params exist
        assert hasattr(network, 'actor_params')
        assert hasattr(network, 'critic_params')
        assert len(network.actor_params) > 0
        assert len(network.critic_params) > 0


class TestCategoricalActorCriticNet:
    """Tests for CategoricalActorCriticNet (discrete action space with value)."""

    def test_categorical_actor_critic_forward_shape(self, set_cpu_device):
        """Test CategoricalActorCriticNet output shapes."""
        state_dim = 4
        action_dim = 2  # CartPole has 2 actions

        actor_body = FCBody(state_dim, (64, 64))
        critic_body = FCBody(state_dim, (64, 64))
        network = CategoricalActorCriticNet(action_dim, actor_body, critic_body)

        batch_size = 10
        obs = torch.randn(batch_size, state_dim)
        output = network(obs)

        # Check all required keys exist
        assert 'action' in output
        assert 'log_pi_a' in output
        assert 'entropy' in output
        assert 'v' in output

        # Check shapes
        assert output['action'].shape == (batch_size,)
        assert output['log_pi_a'].shape == (batch_size, 1)
        assert output['entropy'].shape == (batch_size, 1)
        assert output['v'].shape == (batch_size, 1)

    def test_categorical_actor_critic_action_valid(self, set_cpu_device):
        """Test CategoricalActorCriticNet actions are valid indices."""
        state_dim = 4
        action_dim = 2

        actor_body = FCBody(state_dim, (64, 64))
        critic_body = FCBody(state_dim, (64, 64))
        network = CategoricalActorCriticNet(action_dim, actor_body, critic_body)

        obs = torch.randn(30, state_dim)
        output = network(obs)

        # Actions should be in [0, action_dim)
        assert (output['action'] >= 0).all()
        assert (output['action'] < action_dim).all()

    def test_categorical_actor_critic_params(self, set_cpu_device):
        """Test CategoricalActorCriticNet has separate actor and critic params."""
        state_dim = 4
        action_dim = 2

        actor_body = FCBody(state_dim, (64, 64))
        critic_body = FCBody(state_dim, (64, 64))
        network = CategoricalActorCriticNet(action_dim, actor_body, critic_body)

        assert hasattr(network, 'actor_params')
        assert hasattr(network, 'critic_params')
        assert len(network.actor_params) > 0
        assert len(network.critic_params) > 0