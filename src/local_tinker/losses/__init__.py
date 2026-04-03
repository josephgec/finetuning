"""Loss functions for Local Tinker training."""

from .base import LossFunction
from .cross_entropy import CrossEntropyLoss

__all__ = ["LossFunction", "CrossEntropyLoss"]
