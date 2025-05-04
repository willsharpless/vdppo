import numpy as np
import random
import gym
from gym import spaces
from dm_control import suite

class DMCWrapper(gym.Env):
    metadata = {"render_modes": [], "render_fps": 60}

    def __init__(self, domain_name="cartpole", task_name="balance", max_steps=1000, seed=None):
        self.env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={'random':seed})
        self.max_steps = max_steps
        self.step_count = 0

        # Flattened observation space
        obs_spec = self.env.observation_spec()
        flat_dim = sum(np.prod(v.shape) for v in obs_spec.values())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)

        # Action space from dm_control
        act_spec = self.env.action_spec()
        self.action_space = spaces.Box(act_spec.minimum, act_spec.maximum, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        ts = self.env.reset()
        obs = self._flatten_obs(ts.observation)
        return obs, {}

    def step(self, action):
        self.step_count += 1
        ts = self.env.step(action)
        obs = self._flatten_obs(ts.observation)
        reward = ts.reward if ts.reward is not None else 0.0
        terminated = ts.last()
        truncated = self.step_count >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def _flatten_obs(self, obs_dict):
        return np.concatenate([v.ravel() for v in obs_dict.values()])

    def render(self, height=480, width=640):
        return self.env.physics.render(height=height, width=width, camera_id=0)

## Used with:

# from stable_baselines3 import PPO
# from custom_dmc2gym import DMCWrapper  # adjust import path

# env = DMCWrapper(domain_name="cartpole", task_name="balance")

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=100_000)
