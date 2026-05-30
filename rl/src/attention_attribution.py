"""Attention attribution helper."""
import torch
import torch.nn as nn


class AttentionAttributor:
    """Relevance-style attribution for the actor."""
    def __init__(self, model: nn.Module):
        self.model = model

    def attribute(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        """Score past timesteps for the current action."""
        self.model.zero_grad(set_to_none=True)
        action = self.model(obs_seq, act_seq)

        objective = (action ** 2).sum()
        attn_list = getattr(self.model, "last_attn_weights", None)
        if not attn_list:
            return torch.zeros(obs_seq.size(1), device=obs_seq.device)

        grads = torch.autograd.grad(
            objective,
            attn_list,
            retain_graph=False,
            allow_unused=True,
        )

        relevancies = []
        for attn, grad in zip(attn_list, grads):
            if grad is None:
                continue
            rel = (grad * attn).clamp(min=0.0)
            # Only the current action matters here.
            rel_last = rel[:, :, -1, :]
            relevancies.append(rel_last.mean(dim=1))

        if not relevancies:
            return torch.zeros(obs_seq.size(1), device=obs_seq.device)

        relevance = torch.stack(relevancies, dim=0).mean(dim=0)
        relevance = relevance.mean(dim=0)
        relevance = relevance / relevance.sum().clamp_min(1e-8)
        return relevance
