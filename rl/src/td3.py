"""TD3 implementation."""
import copy
import numpy as np

import torch
import torch.nn as nn


class TD3:
    """Actor-critic wrapper for TD3."""
    def __init__(self, actor, critic1, critic2, cfg):
        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2
        self.actor_target = copy.deepcopy(actor)
        self.critic1_target = copy.deepcopy(critic1)
        self.critic2_target = copy.deepcopy(critic2)

        self.cfg = cfg
        self.gamma = float(cfg.agent.gamma)
        self.tau = float(cfg.agent.tau)
        self.policy_delay = int(cfg.agent.policy_delay)
        self.target_noise = float(cfg.agent.target_noise)
        self.noise_clip = float(cfg.agent.noise_clip)
        self.grad_clip_norm = float(getattr(cfg.agent, "grad_clip_norm", 0.0) or 0.0)

        self.device = next(self.actor.parameters()).device
        lr = float(cfg.agent.lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        self.max_action = float(getattr(actor, "max_action", 1.0))

    def select_action(self, obs, noise: float = 0.0):
        """Select an action, with optional noise."""
        self.actor.eval()
        with torch.no_grad():
            if isinstance(obs, dict):
                obs_seq = torch.as_tensor(obs["obs_seq"], dtype=torch.float32, device=self.device).unsqueeze(0)
                act_seq = torch.as_tensor(obs["act_seq"], dtype=torch.float32, device=self.device).unsqueeze(0)
                action = self.actor(obs_seq, act_seq).squeeze(0)
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = self.actor(obs_t).squeeze(0)

        action = action.cpu().numpy()
        if noise > 0:
            action = action + np.random.normal(0, noise, size=action.shape)
        action = np.clip(action, -self.max_action, self.max_action)
        self.actor.train()
        return action

    def update(self, replay_buffer, batch_size: int, step: int):
        """Run one TD3 update."""
        batch = replay_buffer.sample(batch_size)
        batch = {k: v.to(self.device) for k, v in batch.items()}

        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        has_seq = "obs_seq" in batch

        with torch.no_grad():
            if has_seq:
                next_action = self.actor_target(batch["next_obs_seq"], batch["next_act_seq"])
            else:
                next_action = self.actor_target(next_obs)

            noise = torch.randn_like(next_action) * self.target_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            target_q1 = self.critic1_target(next_obs, next_action)
            target_q2 = self.critic2_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1.0 - done) * self.gamma * target_q

        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        critic1_loss = nn.functional.mse_loss(q1, target_q)
        critic2_loss = nn.functional.mse_loss(q2, target_q)

        self.critic1_optim.zero_grad(set_to_none=True)
        critic1_loss.backward()
        if self.grad_clip_norm > 0.0:
            nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=self.grad_clip_norm)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad(set_to_none=True)
        critic2_loss.backward()
        if self.grad_clip_norm > 0.0:
            nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=self.grad_clip_norm)
        self.critic2_optim.step()

        metrics = {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "actor_loss": 0.0,
        }

        if step % self.policy_delay == 0:
            if has_seq:
                actor_action = self.actor(batch["obs_seq"], batch["act_seq"])
            else:
                actor_action = self.actor(obs)
            actor_loss = -self.critic1(obs, actor_action).mean()

            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.grad_clip_norm > 0.0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip_norm)
            self.actor_optim.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
            metrics["actor_loss"] = float(actor_loss.item())

        return metrics

    def _soft_update(self, net: nn.Module, target_net: nn.Module):
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
