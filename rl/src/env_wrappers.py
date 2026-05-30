"""Hopper wrappers for POMDP tests."""
import numpy as np
import gymnasium as gym


class HiddenVelocityWrapper(gym.ObservationWrapper):
    """Hide Hopper velocity terms."""
    def __init__(self, env):
        super().__init__(env)
        orig = env.observation_space
        keep_dim = 5 if orig.shape[0] >= 11 else max(1, orig.shape[0] // 2)
        self.keep_idx = np.arange(keep_dim)
        self.observation_space = gym.spaces.Box(
            low=orig.low[self.keep_idx],
            high=orig.high[self.keep_idx],
            dtype=orig.dtype,
        )

    def observation(self, obs):
        return obs[self.keep_idx]


class NoisyObservationWrapper(gym.ObservationWrapper):
    """Add Gaussian observation noise."""
    def __init__(self, env, sigma: float = 0.1):
        super().__init__(env)
        self.sigma = float(sigma)

    def observation(self, obs):
        noise = np.random.normal(loc=0.0, scale=self.sigma, size=obs.shape)
        return (obs + noise).astype(np.float32)


class DelayedRewardWrapper(gym.Wrapper):
    """Release reward every K steps."""
    def __init__(self, env, delay_k: int = 10):
        super().__init__(env)
        self.delay_k = int(delay_k)
        self._acc_reward = 0.0
        self._step = 0

    def reset(self, **kwargs):
        self._acc_reward = 0.0
        self._step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._acc_reward += float(reward)
        self._step += 1

        if self._step % self.delay_k == 0 or terminated or truncated:
            out_reward = self._acc_reward
            self._acc_reward = 0.0
        else:
            out_reward = 0.0
        return obs, out_reward, terminated, truncated, info


class CombinedPOMDPWrapper(gym.Wrapper):
    """Combined POMDP wrapper."""
    def __init__(self, env, sigma: float = 0.1, delay_k: int = 10):
        super().__init__(env)
        self.sigma = float(sigma)
        self.delay_k = int(delay_k)
        self._acc_reward = 0.0
        self._step = 0

        orig = env.observation_space
        keep_dim = 5 if orig.shape[0] >= 11 else max(1, orig.shape[0] // 2)
        self.keep_idx = np.arange(keep_dim)
        self.observation_space = gym.spaces.Box(
            low=orig.low[self.keep_idx],
            high=orig.high[self.keep_idx],
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        self._acc_reward = 0.0
        self._step = 0
        obs, info = self.env.reset(**kwargs)
        return self._transform_obs(obs), info

    def _transform_obs(self, obs):
        obs = obs[self.keep_idx]
        noise = np.random.normal(loc=0.0, scale=self.sigma, size=obs.shape)
        return (obs + noise).astype(np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._transform_obs(obs)

        self._acc_reward += float(reward)
        self._step += 1
        if self._step % self.delay_k == 0 or terminated or truncated:
            out_reward = self._acc_reward
            self._acc_reward = 0.0
        else:
            out_reward = 0.0
        return obs, out_reward, terminated, truncated, info


def make_env(cfg) -> gym.Env:
    """Build Hopper with the configured wrapper."""
    env = gym.make(cfg.env.name)
    wrapper = str(cfg.env.wrapper)

    if wrapper == "none":
        return env
    if wrapper == "hidden_velocity":
        return HiddenVelocityWrapper(env)
    if wrapper == "noisy":
        return NoisyObservationWrapper(env, sigma=float(cfg.env.noise_sigma))
    if wrapper == "delayed_reward":
        return DelayedRewardWrapper(env, delay_k=int(cfg.env.delay_k))
    if wrapper == "combined":
        return CombinedPOMDPWrapper(
            env,
            sigma=float(cfg.env.noise_sigma),
            delay_k=int(cfg.env.delay_k),
        )
    raise ValueError(f"Unsupported env wrapper: {wrapper}")
