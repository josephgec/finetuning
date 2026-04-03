"""Tests for utility modules."""

import os
import tempfile
from unittest.mock import patch

import torch

from local_tinker.utils.gpu import estimate_memory, get_gpu_info, track_memory
from local_tinker.utils.logging import CSVLogger, set_logger
from local_tinker.utils.logprobs import get_per_token_logprobs
from local_tinker.utils.tokenizer import apply_chat_template, count_tokens, ensure_pad_token


# ---------------------------------------------------------------------------
# logprobs
# ---------------------------------------------------------------------------


class TestGetPerTokenLogprobs:
    def test_shape(self):
        logits = torch.randn(2, 5, 16)
        token_ids = torch.randint(0, 16, (2, 5))
        result = get_per_token_logprobs(logits, token_ids)
        assert result.shape == (2, 5)

    def test_values_are_log_probs(self):
        logits = torch.randn(1, 3, 8)
        token_ids = torch.randint(0, 8, (1, 3))
        result = get_per_token_logprobs(logits, token_ids)
        assert (result <= 0).all()

    def test_high_logit_gives_high_log_prob(self):
        logits = torch.zeros(1, 1, 8)
        logits[0, 0, 3] = 100.0  # Very high logit for token 3
        token_ids = torch.tensor([[3]])
        result = get_per_token_logprobs(logits, token_ids)
        assert result[0, 0].item() > -0.01  # Should be close to 0


# ---------------------------------------------------------------------------
# GPU utils
# ---------------------------------------------------------------------------


class TestEstimateMemory:
    def test_returns_dict(self):
        result = estimate_memory(1.0)
        assert "model_gb" in result
        assert "lora_gb" in result
        assert "optimizer_gb" in result
        assert "total_gb" in result

    def test_quantize_reduces_model_size(self):
        fp16 = estimate_memory(7.0, quantize=False)
        q4 = estimate_memory(7.0, quantize=True)
        assert q4["model_gb"] < fp16["model_gb"]

    def test_larger_model_needs_more_memory(self):
        small = estimate_memory(1.0)
        large = estimate_memory(8.0)
        assert large["total_gb"] > small["total_gb"]

    def test_higher_rank_uses_more(self):
        low_rank = estimate_memory(7.0, lora_rank=8)
        high_rank = estimate_memory(7.0, lora_rank=64)
        assert high_rank["lora_gb"] > low_rank["lora_gb"]


class TestGetGPUInfo:
    @patch("torch.cuda.is_available", return_value=False)
    def test_no_gpu(self, _):
        assert get_gpu_info() is None


class TestTrackMemory:
    def test_context_manager(self):
        with track_memory() as snapshot:
            _ = torch.randn(100)
        # Snapshot should have zero values on CPU
        assert snapshot.allocated_gb >= 0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestCSVLogger:
    def test_writes_csv(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            logger = CSVLogger(path)
            logger.log({"loss": 2.5, "lr": 1e-4}, step=1)
            logger.log({"loss": 2.0, "lr": 1e-4}, step=2)
            logger.close()

            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 3  # header + 2 rows
            assert "step" in lines[0]
            assert "loss" in lines[0]
        finally:
            os.unlink(path)

    def test_close_idempotent(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            logger = CSVLogger(path)
            logger.log({"loss": 1.0}, step=1)
            logger.close()
            logger.close()  # Should not raise
        finally:
            os.unlink(path)


class TestSetLogger:
    def test_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = set_logger("csv", path=os.path.join(tmpdir, "log.csv"))
            logger.log({"loss": 1.0}, step=1)
            logger.close()

    def test_unknown_backend_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown"):
            set_logger("invalid_backend")


# ---------------------------------------------------------------------------
# Tokenizer utils
# ---------------------------------------------------------------------------


class TestApplyChatTemplate:
    def test_with_apply_method(self):
        class Tok:
            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
                return [1, 2, 3]

        result = apply_chat_template(Tok(), [{"role": "user", "content": "hi"}])
        assert result == [1, 2, 3]

    def test_fallback(self):
        class Tok:
            def __call__(self, text):
                class R:
                    input_ids = [10, 20, 30]
                return R()

        result = apply_chat_template(Tok(), [{"role": "user", "content": "hi"}])
        assert result == [10, 20, 30]


class TestEnsurePadToken:
    def test_sets_pad_token(self):
        class Tok:
            pad_token = None
            eos_token = "<eos>"

        tok = Tok()
        ensure_pad_token(tok)
        assert tok.pad_token == "<eos>"

    def test_preserves_existing(self):
        class Tok:
            pad_token = "<pad>"
            eos_token = "<eos>"

        tok = Tok()
        ensure_pad_token(tok)
        assert tok.pad_token == "<pad>"


class TestCountTokens:
    def test_count(self):
        class Tok:
            def __call__(self, text):
                class R:
                    input_ids = list(range(len(text.split())))
                return R()

        assert count_tokens(Tok(), "hello world test") == 3
