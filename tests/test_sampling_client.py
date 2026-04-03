"""Tests for SamplingClient using mock models (no GPU required)."""

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from local_tinker.sampling_client import SamplingClient
from local_tinker.types import ModelInput, SamplingParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockGenerateOutput:
    """Mimics the dict-style output of model.generate(return_dict_in_generate=True)."""

    def __init__(self, sequences: torch.Tensor, scores: tuple[torch.Tensor, ...]) -> None:
        self.sequences = sequences
        self.scores = scores


class MockGenerativeModel(nn.Module):
    """A fake model whose .generate() returns predetermined tokens."""

    def __init__(self, generated_tokens: list[int], vocab_size: int = 32) -> None:
        super().__init__()
        self._generated_tokens = generated_tokens
        self._vocab_size = vocab_size
        # Need at least one parameter so nn.Module doesn't complain
        self._dummy = nn.Parameter(torch.zeros(1))

    def eval(self):
        return self

    def generate(self, input_ids, **kwargs):
        prompt_len = input_ids.shape[1]
        gen_ids = torch.tensor(self._generated_tokens, dtype=torch.long)
        full_seq = torch.cat([input_ids[0], gen_ids]).unsqueeze(0)

        # Create fake scores (one per generated token)
        scores = []
        for token_id in self._generated_tokens:
            logits = torch.randn(1, self._vocab_size)
            # Make the chosen token have the highest logit
            logits[0, token_id] = 10.0
            scores.append(logits)

        return MockGenerateOutput(
            sequences=full_seq,
            scores=tuple(scores),
        )


class MockTokenizer:
    pad_token_id = 0

    def decode(self, token_ids, skip_special_tokens=False):
        return f"decoded:{','.join(str(t) for t in token_ids)}"


def _make_client(generated_tokens: list[int] | None = None) -> SamplingClient:
    if generated_tokens is None:
        generated_tokens = [10, 11, 12]
    model = MockGenerativeModel(generated_tokens)
    return SamplingClient(
        model=model,
        tokenizer=MockTokenizer(),
        device=torch.device("cpu"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSample:
    def test_returns_sample_response(self):
        sc = _make_client([10, 11, 12])
        prompt = ModelInput.from_ids([1, 2, 3])
        resp = sc.sample(prompt, SamplingParams(max_tokens=10, temperature=0.7))
        assert resp.tokens == [10, 11, 12]
        assert "decoded:" in resp.text
        assert resp.log_probs is not None
        assert len(resp.log_probs) == 3

    def test_default_params(self):
        sc = _make_client([5])
        prompt = ModelInput.from_ids([1])
        # params=None should use defaults
        resp = sc.sample(prompt)
        assert resp.tokens == [5]

    def test_log_probs_are_negative(self):
        sc = _make_client([10, 11])
        prompt = ModelInput.from_ids([1, 2])
        resp = sc.sample(prompt, SamplingParams(temperature=0.7))
        # Log probs should be <= 0
        for lp in resp.log_probs:
            assert lp <= 0

    def test_zero_temperature_branch(self):
        """When temperature=0, do_sample should be False and temperature should be 1.0."""
        model = MockGenerativeModel([7, 8])
        original_generate = model.generate

        called_kwargs = {}

        def capture_generate(input_ids, **kwargs):
            called_kwargs.update(kwargs)
            return original_generate(input_ids, **kwargs)

        model.generate = capture_generate

        sc = SamplingClient(
            model=model,
            tokenizer=MockTokenizer(),
            device=torch.device("cpu"),
        )
        prompt = ModelInput.from_ids([1])
        sc.sample(prompt, SamplingParams(temperature=0.0))

        assert called_kwargs["do_sample"] is False
        assert called_kwargs["temperature"] == 1.0

    def test_positive_temperature_branch(self):
        """When temperature > 0, do_sample should be True."""
        model = MockGenerativeModel([7])
        original_generate = model.generate
        called_kwargs = {}

        def capture_generate(input_ids, **kwargs):
            called_kwargs.update(kwargs)
            return original_generate(input_ids, **kwargs)

        model.generate = capture_generate

        sc = SamplingClient(
            model=model,
            tokenizer=MockTokenizer(),
            device=torch.device("cpu"),
        )
        prompt = ModelInput.from_ids([1])
        sc.sample(prompt, SamplingParams(temperature=0.8))

        assert called_kwargs["do_sample"] is True
        assert called_kwargs["temperature"] == 0.8


class TestBatchSample:
    def test_returns_list(self):
        sc = _make_client([10])
        prompts = [ModelInput.from_ids([1]), ModelInput.from_ids([2])]
        results = sc.batch_sample(prompts, SamplingParams(max_tokens=5))
        assert len(results) == 2
        assert all(r.tokens == [10] for r in results)

    def test_empty_list(self):
        sc = _make_client([10])
        results = sc.batch_sample([], SamplingParams())
        assert results == []


class TestExtractLogProbs:
    def test_basic_extraction(self):
        vocab = 16
        scores = (
            torch.randn(1, vocab),
            torch.randn(1, vocab),
        )
        generated_ids = [3, 7]
        log_probs = SamplingClient._extract_log_probs(scores, generated_ids)
        assert len(log_probs) == 2
        for lp in log_probs:
            assert isinstance(lp, float)
            assert lp <= 0

    def test_more_ids_than_scores(self):
        """If generated_ids is longer than scores, stop at scores length."""
        vocab = 16
        scores = (torch.randn(1, vocab),)
        generated_ids = [3, 7, 11]
        log_probs = SamplingClient._extract_log_probs(scores, generated_ids)
        assert len(log_probs) == 1

    def test_empty_scores(self):
        log_probs = SamplingClient._extract_log_probs((), [1, 2, 3])
        assert log_probs == []

    def test_empty_ids(self):
        scores = (torch.randn(1, 16),)
        log_probs = SamplingClient._extract_log_probs(scores, [])
        assert log_probs == []
