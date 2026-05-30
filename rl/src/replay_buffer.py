"""TD3 replay buffers."""
import numpy as np
import torch


class StandardReplayBuffer:
    """Flat transition replay buffer."""
    def __init__(self, obs_dim: int, act_dim: int, max_size: int = 1_000_000):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, act_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx]),
            "action": torch.as_tensor(self.action[idx]),
            "reward": torch.as_tensor(self.reward[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx]),
            "done": torch.as_tensor(self.done[idx]),
        }


class SequenceReplayBuffer:
    """Trajectory replay with context windows."""
    def __init__(self, obs_dim: int, act_dim: int, context_length: int, max_size: int = 1_000_000):
        self.max_size = int(max_size)
        self.context_length = int(context_length)
        self.ptr = 0
        self.size = 0
        self.current_episode = 0

        self.obs = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, act_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.float32)
        self.episode_id = np.zeros((self.max_size,), dtype=np.int64)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = float(done)
        self.episode_id[self.ptr] = self.current_episode

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        if done:
            self.current_episode += 1

    def sample(self, batch_size: int) -> dict:
        if self.size < 2:
            raise ValueError("Need at least two samples in sequence replay buffer.")

        idx = np.random.randint(0, self.size - 1, size=batch_size)
        obs_seq = []
        act_seq = []
        next_obs_seq = []
        next_act_seq = []

        for i in idx:
            o_s, a_s = self._build_sequence(end_idx=i)
            no_s, na_s = self._build_sequence(end_idx=i + 1)
            obs_seq.append(o_s)
            act_seq.append(a_s)
            next_obs_seq.append(no_s)
            next_act_seq.append(na_s)

        return {
            "obs": torch.as_tensor(self.obs[idx]),
            "action": torch.as_tensor(self.action[idx]),
            "reward": torch.as_tensor(self.reward[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx]),
            "done": torch.as_tensor(self.done[idx]),
            "obs_seq": torch.as_tensor(np.stack(obs_seq, axis=0)),
            "act_seq": torch.as_tensor(np.stack(act_seq, axis=0)),
            "next_obs_seq": torch.as_tensor(np.stack(next_obs_seq, axis=0)),
            "next_act_seq": torch.as_tensor(np.stack(next_act_seq, axis=0)),
        }

    def _build_sequence(self, end_idx: int) -> tuple[np.ndarray, np.ndarray]:
        obs_dim = self.obs.shape[1]
        act_dim = self.action.shape[1]
        obs_seq = np.zeros((self.context_length, obs_dim), dtype=np.float32)
        act_seq = np.zeros((self.context_length, act_dim), dtype=np.float32)

        if end_idx >= self.size:
            return obs_seq, act_seq

        ep = self.episode_id[end_idx]
        start = end_idx - self.context_length + 1
        for pos, idx in enumerate(range(start, end_idx + 1)):
            if idx < 0 or idx >= self.size:
                continue
            if self.episode_id[idx] != ep:
                continue
            obs_seq[pos] = self.obs[idx]
            if idx < end_idx:
                act_seq[pos] = self.action[idx]

        return obs_seq, act_seq
