"""PPO clipped surrogate objective loss."""

from __future__ import annotations

import torch

from .base import LossFunction


class PPOLoss(LossFunction):
    """Proximal Policy Optimization clipped surrogate loss.

    Args:
        clip_range: Clipping parameter epsilon (default 0.2).
        kl_coeff: Coefficient for optional KL penalty against a reference
            model. Pass ``ref_log_probs`` as a kwarg to ``compute`` to enable.
    """

    def __init__(self, clip_range: float = 0.2, kl_coeff: float = 0.0) -> None:
        self.clip_range = clip_range
        self.kl_coeff = kl_coeff

    def compute(
        self, logits: torch.Tensor, labels: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        """Compute PPO clipped surrogate loss.

        Args:
            logits: Current policy logits (batch, seq_len, vocab_size).
            labels: Token IDs used to gather log-probs (batch, seq_len).
            **kwargs:
                old_log_probs: Log-probs under the old policy (batch, seq_len).
                advantages: Per-token advantages (batch, seq_len).
                ref_log_probs: Optional reference model log-probs for KL penalty.

        Returns:
            Scalar loss tensor (negated so minimizing = maximizing reward).
        """
        old_log_probs = kwargs["old_log_probs"]
        advantages = kwargs["advantages"]
        assert isinstance(old_log_probs, torch.Tensor)
        assert isinstance(advantages, torch.Tensor)

        # Current policy log-probs
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


def _gather_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Extract per-token log-probs for the given label tokens.

    Args:
        logits: (batch, seq_len, vocab_size).
        labels: (batch, seq_len) token IDs.

    Returns:
        (batch, seq_len) log-probabilities.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
