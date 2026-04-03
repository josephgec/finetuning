"""Direct Preference Optimization (DPO) loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import LossFunction


class DPOLoss(LossFunction):
    """Direct Preference Optimization loss.

    Computes the DPO objective from chosen/rejected sequence log-probs
    and their reference model counterparts.

    Args:
        beta: Temperature parameter controlling deviation from the reference
            policy (default 0.1).
    """

    def __init__(self, beta: float = 0.1) -> None:
        self.beta = beta

    def compute(
        self, logits: torch.Tensor, labels: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        """Compute DPO loss.

        The batch is expected to contain interleaved chosen/rejected pairs:
        items at even indices are chosen, odd indices are rejected.

        Args:
            logits: Policy logits (batch, seq_len, vocab_size). batch must be
                even — first half chosen, second half rejected.
            labels: Token IDs (batch, seq_len).
            **kwargs:
                ref_chosen_log_probs: (batch/2, seq_len) reference model
                    log-probs for chosen sequences.
                ref_rejected_log_probs: (batch/2, seq_len) reference model
                    log-probs for rejected sequences.

        Returns:
            Scalar DPO loss tensor.
        """
        ref_chosen_log_probs = kwargs["ref_chosen_log_probs"]
        ref_rejected_log_probs = kwargs["ref_rejected_log_probs"]
        assert isinstance(ref_chosen_log_probs, torch.Tensor)
        assert isinstance(ref_rejected_log_probs, torch.Tensor)

        batch = logits.shape[0]
        half = batch // 2

        # Mask padding tokens (labels == -100)
        mask = (labels != -100).float()
        # Clamp labels to valid range for gather (replace -100 with 0)
        safe_labels = labels.clamp(min=0)

        # Current policy log-probs
        log_probs = torch.log_softmax(logits, dim=-1)
        per_token = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        per_token = per_token * mask

        # Sum log-probs per sequence
        policy_chosen = per_token[:half].sum(dim=-1)
        policy_rejected = per_token[half:].sum(dim=-1)

        ref_chosen = (ref_chosen_log_probs * mask[:half]).sum(dim=-1)
        ref_rejected = (ref_rejected_log_probs * mask[half:]).sum(dim=-1)

        # DPO objective
        log_ratio_chosen = policy_chosen - ref_chosen
        log_ratio_rejected = policy_rejected - ref_rejected

        loss = -F.logsigmoid(self.beta * (log_ratio_chosen - log_ratio_rejected)).mean()

        return loss
