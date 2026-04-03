"""Abstract base class for all loss functions."""

from abc import ABC, abstractmethod

import torch


class LossFunction(ABC):
    """Base interface for loss functions used in forward_backward."""

    @abstractmethod
    def compute(
        self, logits: torch.Tensor, labels: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        """Compute a scalar loss with grad_fn attached.

        Args:
            logits: Model output logits of shape (batch, seq_len, vocab_size).
            labels: Target token IDs of shape (batch, seq_len).
                    Use -100 for positions that should be ignored.
            **kwargs: Additional arguments for RL losses (e.g. old_log_probs,
                      advantages, ref_log_probs).

        Returns:
            A scalar tensor with gradient tracking enabled.
        """
        ...
