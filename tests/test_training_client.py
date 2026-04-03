"""Tests for TrainingClient using mock models (no GPU required)."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from local_tinker.losses.base import LossFunction
from local_tinker.losses.cross_entropy import CrossEntropyLoss
from local_tinker.training_client import TrainingClient
from local_tinker.types import AdamParams, Datum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    """Minimal causal LM stand-in with a single linear layer."""

    def __init__(self, vocab_size: int = 32, hidden: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size)
        # Mark all params as trainable (simulates LoRA params)
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, input_ids, attention_mask=None):
        h = self.embed(input_ids)
        logits = self.head(h)

        class Output:
            pass

        out = Output()
        out.logits = logits
        return out

    def train(self, mode=True):
        return super().train(mode)

    def save_pretrained(self, path):
        pass  # no-op for tests


class MockTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, return_tensors=None):
        class R:
            input_ids = torch.tensor([[1, 2, 3]])
        return R()


def _make_client(vocab_size: int = 32) -> TrainingClient:
    model = TinyModel(vocab_size=vocab_size)
    return TrainingClient(
        model=model,
        tokenizer=MockTokenizer(),
        device=torch.device("cpu"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCollate:
    def test_single_datum_no_padding(self):
        tc = _make_client()
        data = [Datum(input_ids=[1, 2, 3], labels=[1, 2, 3])]
        ids, labels, mask = tc._collate(data)
        assert ids.shape == (1, 3)
        assert labels.shape == (1, 3)
        assert mask.shape == (1, 3)
        assert mask.tolist() == [[1, 1, 1]]

    def test_padding_to_max_length(self):
        tc = _make_client()
        data = [
            Datum(input_ids=[1, 2, 3], labels=[1, 2, 3]),
            Datum(input_ids=[4, 5], labels=[4, 5]),
        ]
        ids, labels, mask = tc._collate(data)
        assert ids.shape == (2, 3)
        # Second sequence should be padded with pad_token_id=0
        assert ids[1, 2].item() == 0
        # Labels padded with -100
        assert labels[1, 2].item() == -100
        # Attention mask padded with 0
        assert mask[1, 2].item() == 0

    def test_labels_default_to_input_ids(self):
        tc = _make_client()
        data = [Datum(input_ids=[5, 6, 7])]
        _, labels, _ = tc._collate(data)
        assert labels[0].tolist() == [5, 6, 7]

    def test_custom_attention_mask(self):
        tc = _make_client()
        data = [Datum(input_ids=[1, 2, 3], attention_mask=[1, 0, 1])]
        _, _, mask = tc._collate(data)
        assert mask[0].tolist() == [1, 0, 1]

    def test_custom_attention_mask_padded(self):
        tc = _make_client()
        data = [
            Datum(input_ids=[1, 2, 3], attention_mask=[1, 1, 1]),
            Datum(input_ids=[4], attention_mask=[1]),
        ]
        _, _, mask = tc._collate(data)
        assert mask[1].tolist() == [1, 0, 0]


class TestForwardBackward:
    def test_returns_correct_type(self):
        tc = _make_client()
        data = [Datum(input_ids=[1, 2, 3], labels=[1, 2, 3])]
        result = tc.forward_backward(data, CrossEntropyLoss())
        assert result.loss > 0
        assert result.num_tokens == 3
        assert result.grad_norm is not None
        assert result.grad_norm > 0

    def test_gradients_accumulate(self):
        tc = _make_client()
        data = [Datum(input_ids=[1, 2, 3], labels=[1, 2, 3])]

        # First pass
        tc.forward_backward(data, CrossEntropyLoss())
        grads_after_1 = [
            p.grad.clone() for p in tc._model.parameters()
            if p.requires_grad and p.grad is not None
        ]

        # Second pass — gradients should accumulate (not reset)
        tc.forward_backward(data, CrossEntropyLoss())
        grads_after_2 = [
            p.grad.clone() for p in tc._model.parameters()
            if p.requires_grad and p.grad is not None
        ]

        # At least one gradient should have increased in magnitude
        any_increased = any(
            g2.abs().sum() > g1.abs().sum()
            for g1, g2 in zip(grads_after_1, grads_after_2)
        )
        assert any_increased

    def test_num_tokens_excludes_masked(self):
        tc = _make_client()
        # Labels with some positions masked
        data = [Datum(input_ids=[1, 2, 3, 4], labels=[-100, -100, 3, 4])]
        result = tc.forward_backward(data, CrossEntropyLoss())
        assert result.num_tokens == 2


class TestOptimStep:
    def test_step_increments(self):
        tc = _make_client()
        data = [Datum(input_ids=[1, 2, 3], labels=[1, 2, 3])]
        tc.forward_backward(data, CrossEntropyLoss())

        assert tc.get_step() == 0
        resp = tc.optim_step(AdamParams(lr=1e-3))
        assert resp.step == 1
        assert resp.lr == 1e-3
        assert tc.get_step() == 1

    def test_multiple_steps(self):
        tc = _make_client()
        data = [Datum(input_ids=[1, 2, 3], labels=[1, 2, 3])]

        for i in range(3):
            tc.forward_backward(data, CrossEntropyLoss())
            resp = tc.optim_step(AdamParams(lr=1e-3))
            assert resp.step == i + 1

    def test_lr_change_recreates_optimizer(self):
        tc = _make_client()
        data = [Datum(input_ids=[1, 2, 3], labels=[1, 2, 3])]

        tc.forward_backward(data, CrossEntropyLoss())
        tc.optim_step(AdamParams(lr=1e-3))
        opt1 = tc._optimizer

        tc.forward_backward(data, CrossEntropyLoss())
        tc.optim_step(AdamParams(lr=5e-4))
        opt2 = tc._optimizer

        # Optimizer should have been recreated
        assert opt1 is not opt2

    def test_same_lr_reuses_optimizer(self):
        tc = _make_client()
        data = [Datum(input_ids=[1, 2, 3], labels=[1, 2, 3])]

        tc.forward_backward(data, CrossEntropyLoss())
        tc.optim_step(AdamParams(lr=1e-3))
        opt1 = tc._optimizer

        tc.forward_backward(data, CrossEntropyLoss())
        tc.optim_step(AdamParams(lr=1e-3))
        opt2 = tc._optimizer

        assert opt1 is opt2


class TestGradNorm:
    def test_grad_norm_none_when_no_grads(self):
        tc = _make_client()
        # No forward_backward called, so no grads
        assert tc._grad_norm() is None

    def test_grad_norm_positive_after_backward(self):
        tc = _make_client()
        data = [Datum(input_ids=[1, 2, 3], labels=[1, 2, 3])]
        tc.forward_backward(data, CrossEntropyLoss())
        norm = tc._grad_norm()
        assert norm is not None
        assert norm > 0


class TestSaveLoadWeights:
    def test_save_weights_calls_save_pretrained(self):
        tc = _make_client()
        tc._model.save_pretrained = MagicMock()
        tc.save_weights("/tmp/test_save")
        tc._model.save_pretrained.assert_called_once_with("/tmp/test_save")

    def test_load_weights_non_peft_raises(self):
        tc = _make_client()
        # TinyModel is not a PeftModel, so this should raise
        with pytest.raises(TypeError, match="not a PeftModel"):
            tc.load_weights("/tmp/test_load")

    def test_load_weights_peft_model(self):
        from peft import PeftModel

        tc = _make_client()

        # Replace the model with a MagicMock that passes isinstance(_, PeftModel)
        mock_model = MagicMock(spec=PeftModel)
        tc._model = mock_model

        tc.load_weights("/tmp/test_load")
        mock_model.load_adapter.assert_called_once_with(
            "/tmp/test_load", adapter_name="default"
        )


class TestAutoCheckpoint:
    def test_auto_checkpoint_triggers(self):
        tc = _make_client()
        tc._auto_checkpoint_every = 2
        tc._checkpoint_dir = None  # Will be set below

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tc._checkpoint_dir = tmpdir
            data = [Datum(input_ids=[1, 2, 3], labels=[1, 2, 3])]

            # Step 1: no checkpoint
            tc.forward_backward(data, CrossEntropyLoss())
            tc.optim_step(AdamParams(lr=1e-3))
            ckpts = list(Path(tmpdir).iterdir())
            assert len(ckpts) == 0

            # Step 2: checkpoint fires
            tc.forward_backward(data, CrossEntropyLoss())
            tc.optim_step(AdamParams(lr=1e-3))
            ckpts = [d for d in Path(tmpdir).iterdir() if d.is_dir()]
            assert len(ckpts) == 1

    def test_auto_checkpoint_rotates(self):
        tc = _make_client()
        tc._auto_checkpoint_every = 1
        tc._max_checkpoints = 2

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tc._checkpoint_dir = tmpdir
            data = [Datum(input_ids=[1, 2, 3], labels=[1, 2, 3])]

            for _ in range(5):
                tc.forward_backward(data, CrossEntropyLoss())
                tc.optim_step(AdamParams(lr=1e-3))

            ckpts = [d for d in Path(tmpdir).iterdir() if d.is_dir()]
            assert len(ckpts) <= 2

    def test_no_auto_checkpoint_when_disabled(self):
        tc = _make_client()
        # auto_checkpoint_every defaults to None
        data = [Datum(input_ids=[1, 2, 3], labels=[1, 2, 3])]
        tc.forward_backward(data, CrossEntropyLoss())
        tc.optim_step(AdamParams(lr=1e-3))
        # Should not raise or create checkpoints


class TestGetReferenceLogProbs:
    def test_returns_tensor(self):
        tc = _make_client()
        data = [Datum(input_ids=[1, 2, 3], labels=[1, 2, 3])]
        result = tc.get_reference_log_probs(data)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 3)

    def test_values_are_log_probs(self):
        tc = _make_client()
        data = [Datum(input_ids=[1, 2, 3], labels=[1, 2, 3])]
        result = tc.get_reference_log_probs(data)
        assert (result <= 0).all()

    def test_batch(self):
        tc = _make_client()
        data = [
            Datum(input_ids=[1, 2, 3], labels=[1, 2, 3]),
            Datum(input_ids=[4, 5, 6], labels=[4, 5, 6]),
        ]
        result = tc.get_reference_log_probs(data)
        assert result.shape == (2, 3)


class TestEndToEndTraining:
    def test_loss_decreases_over_steps(self):
        """Verify that repeated training on the same data reduces loss."""
        tc = _make_client()
        data = [Datum(input_ids=[1, 2, 3, 4, 5], labels=[1, 2, 3, 4, 5])]

        losses = []
        for _ in range(10):
            result = tc.forward_backward(data, CrossEntropyLoss())
            tc.optim_step(AdamParams(lr=1e-2))
            losses.append(result.loss)

        # Loss should decrease (first loss > last loss)
        assert losses[0] > losses[-1]
