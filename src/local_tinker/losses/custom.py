"""User-defined custom loss function wrapper."""

from __future__ import annotations

from typing import Callable

import torch

from .base import LossFunction


class CustomLoss(LossFunction):
    """Wrap a user-provided callable as a :class:`LossFunction`.

    Args:
        fn: A callable ``(logits, labels, **kwargs) -> scalar Tensor``.
    """

    def __init__(self, fn: Callable[..., torch.Tensor]) -> None:
        self._fn = fn

    def compute(
        self, logits: torch.Tensor, labels: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        """Delegate to the user-provided callable."""
        return self._fn(logits, labels, **kwargs)
