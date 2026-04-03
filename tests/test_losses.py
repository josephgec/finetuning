"""Tests for all loss functions using small random tensors (no GPU required)."""

import math

import torch

from local_tinker.losses.cross_entropy import CrossEntropyLoss
from local_tinker.losses.ppo import PPOLoss, _gather_log_probs
from local_tinker.losses.grpo import GRPOLoss
from local_tinker.losses.dpo import DPOLoss
from local_tinker.losses.custom import CustomLoss


# ---------------------------------------------------------------------------
# CrossEntropyLoss
# ---------------------------------------------------------------------------


class TestCrossEntropyLoss:
    def _make_logits_and_labels(
        self, batch: int = 2, seq_len: int = 8, vocab: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.randn(batch, seq_len, vocab, requires_grad=True)
        labels = torch.randint(0, vocab, (batch, seq_len))
        return logits, labels

    def test_returns_scalar(self):
        logits, labels = self._make_logits_and_labels()
        loss = CrossEntropyLoss().compute(logits, labels)
        assert loss.ndim == 0

    def test_has_grad_fn(self):
        logits, labels = self._make_logits_and_labels()
        loss = CrossEntropyLoss().compute(logits, labels)
        assert loss.grad_fn is not None

    def test_backward_produces_gradients(self):
        logits, labels = self._make_logits_and_labels()
        loss = CrossEntropyLoss().compute(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_positive_loss(self):
        logits, labels = self._make_logits_and_labels()
        loss = CrossEntropyLoss().compute(logits, labels)
        assert loss.item() > 0

    def test_mask_with_ignore_index(self):
        logits, labels = self._make_logits_and_labels(batch=1, seq_len=10)
        labels[0, :5] = -100
        loss = CrossEntropyLoss().compute(logits, labels)
        assert loss.item() > 0
        loss.backward()
        assert logits.grad is not None

    def test_all_masked_returns_nan(self):
        logits = torch.randn(1, 5, 32, requires_grad=True)
        labels = torch.full((1, 5), -100, dtype=torch.long)
        loss = CrossEntropyLoss().compute(logits, labels)
        assert math.isnan(loss.item())

    def test_perfect_prediction_low_loss(self):
        vocab, seq_len = 16, 4
        logits_data = torch.zeros(1, seq_len, vocab)
        logits_data[:, :, 0] = 100.0
        logits = logits_data.clone().requires_grad_(True)
        labels = torch.zeros(1, seq_len, dtype=torch.long)
        loss = CrossEntropyLoss().compute(logits, labels)
        assert loss.item() < 0.01


# ---------------------------------------------------------------------------
# PPOLoss
# ---------------------------------------------------------------------------


class TestPPOLoss:
    def _make_data(self, batch: int = 4, seq_len: int = 6, vocab: int = 16):
        logits = torch.randn(batch, seq_len, vocab, requires_grad=True)
        labels = torch.randint(0, vocab, (batch, seq_len))
        old_log_probs = torch.randn(batch, seq_len)
        advantages = torch.randn(batch, seq_len)
        return logits, labels, old_log_probs, advantages

    def test_returns_scalar(self):
        logits, labels, old_lp, adv = self._make_data()
        loss = PPOLoss().compute(logits, labels, old_log_probs=old_lp, advantages=adv)
        assert loss.ndim == 0

    def test_has_grad_fn(self):
        logits, labels, old_lp, adv = self._make_data()
        loss = PPOLoss().compute(logits, labels, old_log_probs=old_lp, advantages=adv)
        assert loss.grad_fn is not None

    def test_backward(self):
        logits, labels, old_lp, adv = self._make_data()
        loss = PPOLoss().compute(logits, labels, old_log_probs=old_lp, advantages=adv)
        loss.backward()
        assert logits.grad is not None

    def test_clip_range(self):
        logits, labels, old_lp, adv = self._make_data()
        loss_wide = PPOLoss(clip_range=0.5).compute(logits, labels, old_log_probs=old_lp, advantages=adv)
        loss_narrow = PPOLoss(clip_range=0.1).compute(logits, labels, old_log_probs=old_lp, advantages=adv)
        # Both should produce valid losses
        assert not math.isnan(loss_wide.item())
        assert not math.isnan(loss_narrow.item())

    def test_kl_penalty(self):
        logits, labels, old_lp, adv = self._make_data()
        ref_lp = torch.randn_like(old_lp)
        loss_no_kl = PPOLoss(kl_coeff=0.0).compute(
            logits, labels, old_log_probs=old_lp, advantages=adv
        )
        loss_with_kl = PPOLoss(kl_coeff=0.1).compute(
            logits, labels, old_log_probs=old_lp, advantages=adv, ref_log_probs=ref_lp
        )
        # KL penalty should change the loss
        assert loss_no_kl.item() != loss_with_kl.item()


class TestGatherLogProbs:
    def test_shape(self):
        logits = torch.randn(2, 5, 16)
        labels = torch.randint(0, 16, (2, 5))
        result = _gather_log_probs(logits, labels)
        assert result.shape == (2, 5)

    def test_values_are_log_probs(self):
        logits = torch.randn(1, 3, 8)
        labels = torch.randint(0, 8, (1, 3))
        result = _gather_log_probs(logits, labels)
        # Log-probs should be <= 0
        assert (result <= 0).all()


# ---------------------------------------------------------------------------
# GRPOLoss
# ---------------------------------------------------------------------------


class TestGRPOLoss:
    def test_returns_scalar(self):
        batch, seq_len, vocab = 4, 6, 16
        logits = torch.randn(batch, seq_len, vocab, requires_grad=True)
        labels = torch.randint(0, vocab, (batch, seq_len))
        old_lp = torch.randn(batch, seq_len)
        rewards = torch.randn(batch)

        loss = GRPOLoss().compute(
            logits, labels, old_log_probs=old_lp, rewards=rewards, group_size=4
        )
        assert loss.ndim == 0

    def test_backward(self):
        batch, seq_len, vocab = 4, 6, 16
        logits = torch.randn(batch, seq_len, vocab, requires_grad=True)
        labels = torch.randint(0, vocab, (batch, seq_len))
        old_lp = torch.randn(batch, seq_len)
        rewards = torch.randn(batch)

        loss = GRPOLoss().compute(
            logits, labels, old_log_probs=old_lp, rewards=rewards, group_size=2
        )
        loss.backward()
        assert logits.grad is not None

    def test_group_advantages_normalized(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        adv = GRPOLoss._compute_group_advantages(rewards, group_size=2)
        # Within each group of 2, advantages should be zero-mean
        assert abs(adv[:2].mean().item()) < 1e-5
        assert abs(adv[2:].mean().item()) < 1e-5

    def test_kl_penalty(self):
        batch, seq_len, vocab = 4, 6, 16
        logits = torch.randn(batch, seq_len, vocab, requires_grad=True)
        labels = torch.randint(0, vocab, (batch, seq_len))
        old_lp = torch.randn(batch, seq_len)
        rewards = torch.randn(batch)
        ref_lp = torch.randn(batch, seq_len)

        loss = GRPOLoss(kl_coeff=0.1).compute(
            logits, labels, old_log_probs=old_lp, rewards=rewards,
            group_size=4, ref_log_probs=ref_lp
        )
        assert not math.isnan(loss.item())


# ---------------------------------------------------------------------------
# DPOLoss
# ---------------------------------------------------------------------------


class TestDPOLoss:
    def _make_data(self, half: int = 2, seq_len: int = 6, vocab: int = 16):
        batch = half * 2
        logits = torch.randn(batch, seq_len, vocab, requires_grad=True)
        labels = torch.randint(0, vocab, (batch, seq_len))
        ref_chosen = torch.randn(half, seq_len)
        ref_rejected = torch.randn(half, seq_len)
        return logits, labels, ref_chosen, ref_rejected

    def test_returns_scalar(self):
        logits, labels, rc, rr = self._make_data()
        loss = DPOLoss().compute(
            logits, labels, ref_chosen_log_probs=rc, ref_rejected_log_probs=rr
        )
        assert loss.ndim == 0

    def test_has_grad_fn(self):
        logits, labels, rc, rr = self._make_data()
        loss = DPOLoss().compute(
            logits, labels, ref_chosen_log_probs=rc, ref_rejected_log_probs=rr
        )
        assert loss.grad_fn is not None

    def test_backward(self):
        logits, labels, rc, rr = self._make_data()
        loss = DPOLoss().compute(
            logits, labels, ref_chosen_log_probs=rc, ref_rejected_log_probs=rr
        )
        loss.backward()
        assert logits.grad is not None

    def test_beta_affects_loss(self):
        logits, labels, rc, rr = self._make_data()
        loss_low = DPOLoss(beta=0.01).compute(
            logits, labels, ref_chosen_log_probs=rc, ref_rejected_log_probs=rr
        )
        loss_high = DPOLoss(beta=1.0).compute(
            logits, labels, ref_chosen_log_probs=rc, ref_rejected_log_probs=rr
        )
        assert loss_low.item() != loss_high.item()

    def test_handles_masked_labels(self):
        logits, labels, rc, rr = self._make_data()
        labels[:, :2] = -100  # mask first 2 tokens
        loss = DPOLoss().compute(
            logits, labels, ref_chosen_log_probs=rc, ref_rejected_log_probs=rr
        )
        assert not math.isnan(loss.item())


# ---------------------------------------------------------------------------
# CustomLoss
# ---------------------------------------------------------------------------


class TestCustomLoss:
    def test_delegates_to_callable(self):
        def my_loss(logits, labels, **kwargs):
            return logits.sum() * 0 + 42.0

        loss_fn = CustomLoss(my_loss)
        logits = torch.randn(2, 4, 16)
        labels = torch.randint(0, 16, (2, 4))
        loss = loss_fn.compute(logits, labels)
        assert loss.item() == 42.0

    def test_passes_kwargs(self):
        received_kwargs = {}

        def my_loss(logits, labels, **kwargs):
            received_kwargs.update(kwargs)
            return logits.mean()

        loss_fn = CustomLoss(my_loss)
        logits = torch.randn(2, 4, 16, requires_grad=True)
        labels = torch.randint(0, 16, (2, 4))
        loss_fn.compute(logits, labels, extra_param="hello")
        assert received_kwargs["extra_param"] == "hello"

    def test_backward(self):
        def my_loss(logits, labels, **kwargs):
            return logits.mean()

        logits = torch.randn(2, 4, 16, requires_grad=True)
        labels = torch.randint(0, 16, (2, 4))
        loss = CustomLoss(my_loss).compute(logits, labels)
        loss.backward()
        assert logits.grad is not None
