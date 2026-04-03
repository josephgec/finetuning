"""Tests for loss functions using small random tensors (no GPU required)."""

import torch

from local_tinker.losses.cross_entropy import CrossEntropyLoss


class TestCrossEntropyLoss:
    def _make_logits_and_labels(
        self, batch: int = 2, seq_len: int = 8, vocab: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create random logits and label tensors for testing."""
        logits = torch.randn(batch, seq_len, vocab, requires_grad=True)
        labels = torch.randint(0, vocab, (batch, seq_len))
        return logits, labels

    def test_returns_scalar(self):
        logits, labels = self._make_logits_and_labels()
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.compute(logits, labels)
        assert loss.ndim == 0  # scalar

    def test_has_grad_fn(self):
        logits, labels = self._make_logits_and_labels()
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.compute(logits, labels)
        assert loss.grad_fn is not None

    def test_backward_produces_gradients(self):
        logits, labels = self._make_logits_and_labels()
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.compute(logits, labels)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

    def test_positive_loss(self):
        logits, labels = self._make_logits_and_labels()
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.compute(logits, labels)
        assert loss.item() > 0

    def test_mask_with_ignore_index(self):
        """Labels set to -100 should be ignored."""
        logits, labels = self._make_logits_and_labels(batch=1, seq_len=10)
        # Mask out some positions
        labels[0, :5] = -100
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.compute(logits, labels)
        assert loss.item() > 0
        # Should still produce gradients
        loss.backward()
        assert logits.grad is not None

    def test_all_masked_returns_nan(self):
        """If all labels are -100, PyTorch cross_entropy returns nan (no valid targets)."""
        import math

        logits = torch.randn(1, 5, 32, requires_grad=True)
        labels = torch.full((1, 5), -100, dtype=torch.long)
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.compute(logits, labels)
        assert math.isnan(loss.item())

    def test_perfect_prediction_low_loss(self):
        """When logits strongly predict the correct token, loss should be low."""
        vocab = 16
        seq_len = 4
        logits = torch.zeros(1, seq_len, vocab, requires_grad=True)
        labels = torch.zeros(1, seq_len, dtype=torch.long)

        # Make logits strongly predict token 0 at every position
        with torch.no_grad():
            logits_data = torch.zeros(1, seq_len, vocab)
            logits_data[:, :, 0] = 100.0  # very high logit for token 0

        logits = logits_data.clone().requires_grad_(True)
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.compute(logits, labels)
        assert loss.item() < 0.01
