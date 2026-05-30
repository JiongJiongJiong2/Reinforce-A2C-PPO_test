import gymnasium as gym
from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete

from utils import *


# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(env_id, seed):
    random_seed(seed)
    env = gym.make(env_id)
    env = OriginalReturnWrapper(env, seed)
    return env


class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env, seed=None):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0
        self.seed = seed

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.total_rewards += reward
        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
            obs, _ = self.env.reset(seed=self.seed)
        else:
            info['episodic_return'] = None
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs



class Task:
    def __init__(self,
                 name,
                 seed=None):

        if seed is None:
            seed = np.random.randint(int(1e9))

        self.env = make_env(name, seed)
        self.name = name
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)