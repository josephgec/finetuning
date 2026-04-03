"""Standard causal language model cross-entropy loss."""

import torch
import torch.nn.functional as F

from .base import LossFunction


class CrossEntropyLoss(LossFunction):
    """Next-token prediction cross-entropy loss.

    Shifts logits and labels internally so the caller passes unshifted
    sequences (matching the HuggingFace convention).

    Args:
        mask_prompt_tokens: If True, positions where labels == -100 are
            ignored. Callers can set prompt positions to -100 in their
            Datum.labels to only train on completion tokens.
    """

    def __init__(self, mask_prompt_tokens: bool = True) -> None:
        self.mask_prompt_tokens = mask_prompt_tokens

    def compute(
        self, logits: torch.Tensor, labels: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        """Compute causal LM cross-entropy loss.

        Args:
            logits: (batch, seq_len, vocab_size) — raw model output.
            labels: (batch, seq_len) — target token IDs. Use -100 to mask.

        Returns:
            Scalar loss tensor.
        """
        # Shift so that token n predicts token n+1
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten for cross_entropy
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss
