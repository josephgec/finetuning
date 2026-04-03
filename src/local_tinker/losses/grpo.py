"""Group Relative Policy Optimization (GRPO) loss."""

from __future__ import annotations

import torch

from .base import LossFunction
from .ppo import _gather_log_probs


class GRPOLoss(LossFunction):
    """Group Relative Policy Optimization loss.

    Like PPO but computes advantages from group-relative rewards: for each
    prompt, sample N completions, score them, and normalize rewards within
    the group.

    Args:
        clip_range: Clipping parameter epsilon (default 0.2).
        kl_coeff: Coefficient for optional KL penalty.
    """

    def __init__(self, clip_range: float = 0.2, kl_coeff: float = 0.0) -> None:
        self.clip_range = clip_range
        self.kl_coeff = kl_coeff

    def compute(
        self, logits: torch.Tensor, labels: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        """Compute GRPO loss.

        Args:
            logits: Current policy logits (batch, seq_len, vocab_size).
            labels: Token IDs (batch, seq_len).
            **kwargs:
                old_log_probs: (batch, seq_len) log-probs under old policy.
                rewards: (batch,) scalar rewards per completion.
                group_size: Number of completions per prompt. batch must be
                    divisible by group_size.
                ref_log_probs: Optional reference log-probs for KL penalty.

        Returns:
            Scalar loss tensor.
        """
        old_log_probs = kwargs["old_log_probs"]
        rewards = kwargs["rewards"]
        group_size = int(kwargs.get("group_size", logits.shape[0]))  # type: ignore[arg-type]
        assert isinstance(old_log_probs, torch.Tensor)
        assert isinstance(rewards, torch.Tensor)

        # Compute group-relative advantages
        advantages = self._compute_group_advantages(rewards, group_size)

        # Expand advantages to per-token
        advantages = advantages.unsqueeze(-1).expand_as(old_log_probs)

        # Current log-probs
        curr_log_probs = _gather_log_probs(logits, labels)

        ratio = torch.exp(curr_log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        surrogate = torch.min(ratio * advantages, clipped * advantages)
        loss = -surrogate.mean()

        # Optional KL penalty
        if self.kl_coeff > 0 and "ref_log_probs" in kwargs:
            ref_log_probs = kwargs["ref_log_probs"]
            assert isinstance(ref_log_probs, torch.Tensor)
            kl = (curr_log_probs - ref_log_probs).mean()
            loss = loss + self.kl_coeff * kl

        return loss

    @staticmethod
    def _compute_group_advantages(
        rewards: torch.Tensor, group_size: int
    ) -> torch.Tensor:
        """Normalize rewards within groups to produce advantages.

        Args:
            rewards: (batch,) scalar rewards.
            group_size: Number of completions per prompt group.

        Returns:
            (batch,) normalized advantages.
        """
        batch_size = rewards.shape[0]
        num_groups = batch_size // group_size
        grouped = rewards.view(num_groups, group_size)

        mean = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, keepdim=True)
        # Avoid division by zero when all rewards in a group are the same
        std = std.clamp(min=1e-8)
        normalized = (grouped - mean) / std

        return normalized.view(batch_size)
