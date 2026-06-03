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
        self.target_noise_final = float(getattr(cfg.agent, "target_noise_final", self.target_noise))
        self.target_noise_decay_steps = int(getattr(cfg.agent, "target_noise_decay_steps", 0) or 0)
        self.noise_clip_final = float(getattr(cfg.agent, "noise_clip_final", self.noise_clip))
        self.noise_clip_decay_steps = int(getattr(cfg.agent, "noise_clip_decay_steps", 0) or 0)
        self.start_steps = int(getattr(cfg.agent, "start_steps", 0) or 0)
        self.critic_loss_type = str(getattr(cfg.agent, "critic_loss", "mse")).lower()
        self.huber_beta = float(getattr(cfg.agent, "huber_beta", 1.0) or 1.0)
        self.target_q_clip = float(getattr(cfg.agent, "target_q_clip", 0.0) or 0.0)

        self.device = next(self.actor.parameters()).device
        lr = float(cfg.agent.lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        self.max_action = float(getattr(actor, "max_action", 1.0))

    def _scheduled_value(self, initial: float, final: float, decay_steps: int, step: int) -> float:
        if int(decay_steps) <= 0:
            return float(initial)
        progress = (int(step) - self.start_steps) / float(decay_steps)
        progress = float(np.clip(progress, 0.0, 1.0))
        return float(initial + progress * (final - initial))

    def _critic_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.critic_loss_type == "mse":
            return nn.functional.mse_loss(pred, target)
        if self.critic_loss_type == "huber":
            return nn.functional.smooth_l1_loss(pred, target, beta=self.huber_beta)
        raise ValueError(f"Unsupported critic_loss: {self.critic_loss_type}")

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
        target_noise = self._scheduled_value(
            self.target_noise,
            self.target_noise_final,
            self.target_noise_decay_steps,
            step,
        )
        noise_clip = self._scheduled_value(
            self.noise_clip,
            self.noise_clip_final,
            self.noise_clip_decay_steps,
            step,
        )

        with torch.no_grad():
            if has_seq:
                next_action = self.actor_target(batch["next_obs_seq"], batch["next_act_seq"])
            else:
                next_action = self.actor_target(next_obs)

            noise = torch.randn_like(next_action) * target_noise
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            target_q1 = self.critic1_target(next_obs, next_action)
            target_q2 = self.critic2_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1.0 - done) * self.gamma * target_q
            unclipped_target_q = target_q
            if self.target_q_clip > 0.0:
                target_q = target_q.clamp(-self.target_q_clip, self.target_q_clip)
                target_q_clipped_frac = (target_q != unclipped_target_q).float().mean()
            else:
                target_q_clipped_frac = torch.zeros((), device=self.device)

        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        critic1_loss = self._critic_loss(q1, target_q)
        critic2_loss = self._critic_loss(q2, target_q)

        self.critic1_optim.zero_grad(set_to_none=True)
        critic1_loss.backward()
        critic1_grad_norm = float("nan")
        if self.grad_clip_norm > 0.0:
            critic1_grad_norm = float(
                nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=self.grad_clip_norm).item()
            )
        self.critic1_optim.step()

        self.critic2_optim.zero_grad(set_to_none=True)
        critic2_loss.backward()
        critic2_grad_norm = float("nan")
        if self.grad_clip_norm > 0.0:
            critic2_grad_norm = float(
                nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=self.grad_clip_norm).item()
            )
        self.critic2_optim.step()

        metrics = {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "actor_loss": 0.0,
            "critic_loss_type": self.critic_loss_type,
            "critic1_grad_norm": critic1_grad_norm,
            "critic2_grad_norm": critic2_grad_norm,
            "actor_grad_norm": float("nan"),
            "target_noise_std": float(target_noise),
            "target_noise_clip": float(noise_clip),
            "target_q_mean": float(target_q.mean().item()),
            "target_q_abs_max": float(target_q.abs().max().item()),
            "target_q_clipped_frac": float(target_q_clipped_frac.item()),
            "q1_mean": float(q1.mean().item()),
            "q2_mean": float(q2.mean().item()),
            "q_gap_abs_mean": float((q1 - q2).abs().mean().item()),
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
                metrics["actor_grad_norm"] = float(
                    nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip_norm).item()
                )
            self.actor_optim.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
            metrics["actor_loss"] = float(actor_loss.item())

        return metrics

    def _soft_update(self, net: nn.Module, target_net: nn.Module):
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
