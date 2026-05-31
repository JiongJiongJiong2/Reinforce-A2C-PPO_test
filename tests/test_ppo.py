"""
Tests for PPO module in section_2/ppo.py
"""
import pytest
import torch
import numpy as np

from project_2_code.section_2.ppo import (
    set_seed,
    MDPModel,
    ReplayMemory,
    Transition
)


class TestSetSeed:
    """Tests for set_seed function."""

    def test_set_seed_reproducibility(self):
        """Test set_seed produces reproducible random numbers."""
        import random

        set_seed(42)
        r1 = random.random()
        n1 = np.random.rand()
        t1 = torch.rand(1).item()

        set_seed(42)
        r2 = random.random()
        n2 = np.random.rand()
        t2 = torch.rand(1).item()

        assert r1 == r2
        assert n1 == n2
        assert t1 == t2


class TestReplayMemory:
    """Tests for ReplayMemory class."""

    def test_replay_memory_init(self):
        """Test ReplayMemory initialization."""
        memory = ReplayMemory(capacity=100)
        assert len(memory) == 0

    def test_replay_memory_push(self):
        """Test ReplayMemory push operation."""
        memory = ReplayMemory(capacity=100)

        state = torch.randn(4)
        action = torch.tensor(0)
        reward = torch.tensor(1.0)
        next_state = torch.randn(4)
        terminated = torch.tensor(0)

        memory.push(state, action, reward, next_state, terminated)
        assert len(memory) == 1

    def test_replay_memory_capacity(self):
        """Test ReplayMemory respects capacity limit."""
        capacity = 10
        memory = ReplayMemory(capacity=capacity)

        for i in range(20):
            state = torch.randn(4)
            action = torch.tensor(i % 2)
            reward = torch.tensor(1.0)
            next_state = torch.randn(4)
            terminated = torch.tensor(0)
            memory.push(state, action, reward, next_state, terminated)

        assert len(memory) == capacity

    def test_replay_memory_sample(self):
        """Test ReplayMemory sample operation."""
        memory = ReplayMemory(capacity=100)

        # Add some transitions
        for i in range(50):
            state = torch.randn(4)
            action = torch.tensor(i % 2)
            reward = torch.tensor(1.0)
            next_state = torch.randn(4)
            terminated = torch.tensor(0)
            memory.push(state, action, reward, next_state, terminated)

        batch_size = 10
        samples = memory.sample(batch_size)
        assert len(samples) == batch_size

        # Each sample should be a Transition
        for sample in samples:
            assert isinstance(sample, Transition)


class TestMDPModel:
    """Tests for MDPModel (PPO network)."""

    def _make_ppo_args(self):
        """Create minimal args for PPO testing."""
        import argparse
        args = argparse.Namespace()
        args.algorithm = 'PPO'
        args.seed = 42
        args.figure = None
        return args

    def test_mdp_model_init(self):
        """Test MDPModel initialization for PPO."""
        args = self._make_ppo_args()
        state_dim = 4
        n_actions = 2

        model = MDPModel(state_dim, n_actions, args)
        assert model.dim == state_dim
        assert model.n_actions == n_actions
        assert model.algorithm == 'PPO'

    def test_mdp_model_forward_shape(self):
        """Test MDPModel forward output shape for PPO."""
        args = self._make_ppo_args()
        state_dim = 4
        n_actions = 2

        model = MDPModel(state_dim, n_actions, args)
        batch_size = 8
        x = torch.randn(batch_size, state_dim)

        output = model(x)
        # PPO output should be softmax probabilities
        assert output.shape == (batch_size, n_actions)

    def test_mdp_model_forward_probabilities(self):
        """Test MDPModel forward output is valid probabilities for PPO."""
        args = self._make_ppo_args()
        state_dim = 4
        n_actions = 2

        model = MDPModel(state_dim, n_actions, args)
        x = torch.randn(10, state_dim)

        output = model(x)
        # Should be probabilities (sum to 1, all positive)
        assert (output >= 0).all()
        assert torch.allclose(output.sum(dim=1), torch.ones(10))

    def test_mdp_model_forward_baseline_shape(self):
        """Test MDPModel forward_baseline output shape."""
        args = self._make_ppo_args()
        state_dim = 4
        n_actions = 2

        model = MDPModel(state_dim, n_actions, args)
        batch_size = 8
        x = torch.randn(batch_size, state_dim)

        v = model.forward_baseline(x)
        assert v.shape == (batch_size, 1)

    def test_mdp_model_act_ppo(self):
        """Test MDPModel act method for PPO."""
        args = self._make_ppo_args()
        state_dim = 4
        n_actions = 2

        model = MDPModel(state_dim, n_actions, args)
        state = torch.randn(1, state_dim)

        probs = model.act(state, iteration=0)
        assert probs.shape == (1, n_actions)
        assert (probs >= 0).all()

    def test_mdp_model_get_state_action_values(self):
        """Test MDPModel get_state_action_values."""
        args = self._make_ppo_args()
        state_dim = 4
        n_actions = 2

        model = MDPModel(state_dim, n_actions, args)
        batch_state = torch.randn(5, state_dim)
        batch_action = torch.tensor([0, 1, 0, 1, 0])

        values = model.get_state_action_values(batch_state, batch_action)
        assert values.shape == (5,)

    def test_mdp_model_update_batch_ppo(self):
        """Test MDPModel update_batch for PPO algorithm."""
        args = self._make_ppo_args()
        state_dim = 4
        n_actions = 2
        N = 16  # small batch for testing

        model = MDPModel(state_dim, n_actions, args)

        # Create a small batch
        batch_state = torch.randn(N, state_dim)
        batch_action = torch.randint(0, n_actions, (N,))
        batch_reward = torch.randn(N)
        batch_next_state = torch.randn(N, state_dim)
        batch_terminated = torch.zeros(N, dtype=torch.int)
        batch_action_prob = torch.rand(N)  # old policy probabilities

        # Perform update - should not raise any error
        model.update_batch(
            batch_state=batch_state,
            batch_action=batch_action,
            batch_reward=batch_reward,
            batch_next_state=batch_next_state,
            batch_terminated=batch_terminated,
            batch_action_prob=batch_action_prob,
            memory=None,
            target_net=None
        )

        # Just verify it runs without error
        assert True


class TestPPOIntegration:
    """Integration tests for PPO."""

    def test_ppo_episode_interaction(self):
        """Test PPO can interact with CartPole environment."""
        import gymnasium as gym

        env = gym.make('CartPole-v1')
        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        import argparse
        args = argparse.Namespace()
        args.algorithm = 'PPO'
        args.seed = 42
        args.figure = None

        set_seed(42)
        model = MDPModel(state_dim, n_actions, args)

        state, _ = env.reset()
        state = torch.tensor(state).unsqueeze(0)

        # Get action probabilities
        probs = model.act(state, iteration=0)

        # Sample action
        action = torch.multinomial(probs, num_samples=1).item()

        # Step environment
        next_state, reward, terminated, truncated, _ = env.step(action)

        assert isinstance(action, int)
        assert action in [0, 1]
        env.close()

    def test_ppo_short_training(self):
        """Test PPO can run a short training loop."""
        import gymnasium as gym

        env = gym.make('CartPole-v1')
        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        import argparse
        args = argparse.Namespace()
        args.algorithm = 'PPO'
        args.seed = 42
        args.figure = None

        # PPO parameters (small for testing)
        GAMMA = 0.99
        N = 32  # batch size
        MINIBATCH_SIZE = 16
        EPS = 0.1

        set_seed(42)
        model = MDPModel(state_dim, n_actions, args)

        # Collect a small batch
        batch_state = []
        batch_action = []
        batch_reward = []
        batch_next_state = []
        batch_terminated = []
        batch_action_prob = []

        state, _ = env.reset()

        for _ in range(N):
            state_tensor = torch.tensor(state).unsqueeze(0)
            probs = model.act(state_tensor, iteration=0)
            action = torch.multinomial(probs, num_samples=1).item()
            action_prob = probs[0, action].item()

            next_state, reward, terminated, truncated, _ = env.step(action)

            batch_state.append(state_tensor)
            batch_action.append(torch.as_tensor(action))
            batch_reward.append(torch.as_tensor(reward))
            batch_next_state.append(torch.as_tensor(next_state))
            batch_terminated.append(torch.as_tensor(int(terminated)))
            batch_action_prob.append(torch.as_tensor(action_prob))

            if terminated or truncated:
                state, _ = env.reset()
            else:
                state = next_state

        # Convert to tensors
        batch_state = torch.cat(batch_state, dim=0)
        batch_action = torch.stack(batch_action)
        batch_reward = torch.stack(batch_reward)
        batch_next_state = torch.stack(batch_next_state)
        batch_terminated = torch.stack(batch_terminated)
        batch_action_prob = torch.stack(batch_action_prob)

        # Perform update
        model.update_batch(
            batch_state=batch_state,
            batch_action=batch_action,
            batch_reward=batch_reward,
            batch_next_state=batch_next_state,
            batch_terminated=batch_terminated,
            batch_action_prob=batch_action_prob,
            memory=None,
            target_net=None
        )

        # Should complete without error
        env.close()
        assert True