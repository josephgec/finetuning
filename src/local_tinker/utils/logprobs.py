"""Per-token log-probability extraction utilities."""

from __future__ import annotations

import torch


def get_per_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    """Extract per-token log-probabilities from model logits.

    Args:
        logits: (batch, seq_len, vocab_size) raw model output.
        token_ids: (batch, seq_len) target token IDs.

    Returns:
        (batch, seq_len) tensor of log-probabilities.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
