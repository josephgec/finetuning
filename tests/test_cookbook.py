"""Tests for cookbook modules."""

import json
import os
import tempfile

import pytest
import torch

from local_tinker.types import Datum, SampleResponse


# ---------------------------------------------------------------------------
# Mock tokenizer used across tests
# ---------------------------------------------------------------------------


class MockTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"

    def __call__(self, text, return_tensors=None, add_special_tokens=True, **kwargs):
        # Simple word-level tokenizer
        ids = [hash(w) % 100 for w in text.split()]
        if return_tensors == "pt":
            class R:
                input_ids = torch.tensor([ids])
            return R()

        class R:
            input_ids = ids
        return R()

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)


# ---------------------------------------------------------------------------
# SupervisedDataset
# ---------------------------------------------------------------------------


class TestSupervisedDataset:
    def test_from_examples(self):
        from cookbook.supervised import SupervisedDataset

        examples = [
            {"input": "hello", "output": " world"},
            {"input": "foo", "output": " bar"},
        ]
        ds = SupervisedDataset(examples, MockTokenizer())
        assert len(ds) == 2

    def test_iter(self):
        from cookbook.supervised import SupervisedDataset

        examples = [{"input": "a", "output": " b"}]
        ds = SupervisedDataset(examples, MockTokenizer())
        items = list(ds)
        assert len(items) == 1
        assert isinstance(items[0], Datum)

    def test_getitem(self):
        from cookbook.supervised import SupervisedDataset

        examples = [{"input": "a", "output": " b"}]
        ds = SupervisedDataset(examples, MockTokenizer())
        item = ds[0]
        assert isinstance(item, Datum)

    def test_batch(self):
        from cookbook.supervised import SupervisedDataset

        examples = [{"input": f"q{i}", "output": f" a{i}"} for i in range(5)]
        ds = SupervisedDataset(examples, MockTokenizer())
        batches = list(ds.batch(2))
        assert len(batches) == 3  # 2 + 2 + 1
        assert len(batches[0]) == 2
        assert len(batches[2]) == 1

    def test_from_jsonl(self):
        from cookbook.supervised import SupervisedDataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"input": "hello", "output": " world"}) + "\n")
            f.write(json.dumps({"input": "foo", "output": " bar"}) + "\n")
            path = f.name

        try:
            ds = SupervisedDataset.from_jsonl(path, MockTokenizer())
            assert len(ds) == 2
        finally:
            os.unlink(path)

    def test_labels_mask_prompt(self):
        from cookbook.supervised import SupervisedDataset

        examples = [{"input": "prompt", "output": " completion"}]
        ds = SupervisedDataset(examples, MockTokenizer())
        datum = ds[0]
        # First tokens (prompt) should be masked
        assert datum.labels[0] == -100


# ---------------------------------------------------------------------------
# ChatDatasetBuilder
# ---------------------------------------------------------------------------


class TestChatDatasetBuilder:
    def test_build(self):
        from cookbook.supervised import ChatDatasetBuilder

        conversations = [
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        ]
        builder = ChatDatasetBuilder(MockTokenizer())
        data = builder.build(conversations)
        assert len(data) == 1
        assert isinstance(data[0], Datum)

    def test_train_on_all(self):
        from cookbook.supervised import ChatDatasetBuilder

        conversations = [
            [
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "response"},
            ]
        ]
        builder = ChatDatasetBuilder(MockTokenizer(), train_on_assistant_only=False)
        data = builder.build(conversations)
        # When not masking, labels should equal input_ids
        assert data[0].labels == data[0].input_ids


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


class TestRenderers:
    def test_get_renderer_llama(self):
        from cookbook.renderers import get_renderer, Llama3Renderer

        renderer = get_renderer("meta-llama/Llama-3.2-1B-Instruct")
        assert isinstance(renderer, Llama3Renderer)

    def test_get_renderer_qwen(self):
        from cookbook.renderers import get_renderer, QwenRenderer

        renderer = get_renderer("Qwen/Qwen2.5-7B-Instruct")
        assert isinstance(renderer, QwenRenderer)

    def test_get_renderer_mistral(self):
        from cookbook.renderers import get_renderer, MistralRenderer

        renderer = get_renderer("mistralai/Mistral-7B-Instruct-v0.3")
        assert isinstance(renderer, MistralRenderer)

    def test_get_renderer_unknown_raises(self):
        from cookbook.renderers import get_renderer

        with pytest.raises(ValueError, match="No renderer"):
            get_renderer("unknown/model-xyz")

    def test_llama3_render(self):
        from cookbook.renderers import Llama3Renderer

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        renderer = Llama3Renderer()
        datum = renderer.render(messages, MockTokenizer())
        assert isinstance(datum, Datum)
        assert len(datum.input_ids) > 0
        assert len(datum.labels) == len(datum.input_ids)

    def test_qwen_render(self):
        from cookbook.renderers import QwenRenderer

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        datum = QwenRenderer().render(messages, MockTokenizer())
        assert isinstance(datum, Datum)

    def test_mistral_render(self):
        from cookbook.renderers import MistralRenderer

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        datum = MistralRenderer().render(messages, MockTokenizer())
        assert isinstance(datum, Datum)


# ---------------------------------------------------------------------------
# RL environments
# ---------------------------------------------------------------------------


class TestMessageEnv:
    def test_reset_and_step(self):
        from cookbook.rl import MessageEnv

        env = MessageEnv("What is 2+2?", reward_fn=lambda x: 1.0 if "4" in x else 0.0)
        obs = env.reset()
        assert obs == "What is 2+2?"

        _, reward, done, _ = env.step("The answer is 4")
        assert reward == 1.0
        assert done is True

    def test_wrong_answer(self):
        from cookbook.rl import MessageEnv

        env = MessageEnv("What is 2+2?", reward_fn=lambda x: 1.0 if "4" in x else 0.0)
        env.reset()
        _, reward, _, _ = env.step("The answer is 5")
        assert reward == 0.0


class TestProblemEnv:
    def test_correct_answer(self):
        from cookbook.rl import ProblemEnv

        env = ProblemEnv([{"question": "1+1?", "answer": "2"}])
        obs = env.reset()
        assert obs == "1+1?"

        _, reward, done, info = env.step("2")
        assert reward == 1.0
        assert done is True
        assert info["correct"] is True

    def test_wrong_answer(self):
        from cookbook.rl import ProblemEnv

        env = ProblemEnv([{"question": "1+1?", "answer": "2"}])
        env.reset()
        _, reward, _, info = env.step("3")
        assert reward == 0.0
        assert info["correct"] is False

    def test_cycles_through_problems(self):
        from cookbook.rl import ProblemEnv

        problems = [
            {"question": "a?", "answer": "1"},
            {"question": "b?", "answer": "2"},
        ]
        env = ProblemEnv(problems)
        assert env.reset() == "a?"
        env.step("1")
        assert env.reset() == "b?"
        env.step("2")
        # Should cycle
        assert env.reset() == "a?"

    def test_custom_extract(self):
        from cookbook.rl import ProblemEnv

        env = ProblemEnv(
            [{"question": "q", "answer": "42"}],
            extract_answer_fn=lambda x: x.split("=")[-1].strip(),
        )
        env.reset()
        _, reward, _, _ = env.step("answer = 42")
        assert reward == 1.0


class TestTrajectory:
    def test_total_reward(self):
        from cookbook.rl import Trajectory, TrajectoryStep

        t = Trajectory(steps=[
            TrajectoryStep("obs", "act1", 1.0, -0.5),
            TrajectoryStep("obs", "act2", 0.5, -0.3),
        ])
        assert t.total_reward == 1.5

    def test_actions(self):
        from cookbook.rl import Trajectory, TrajectoryStep

        t = Trajectory(steps=[
            TrajectoryStep("obs", "hello", 1.0, -0.5),
            TrajectoryStep("obs", "world", 0.5, -0.3),
        ])
        assert t.actions == ["hello", "world"]


class TestComputeAdvantages:
    def test_returns_data(self):
        from cookbook.rl import Trajectory, TrajectoryStep, compute_advantages

        trajectories = [
            Trajectory(steps=[TrajectoryStep("q", "a1", 1.0, -0.5)]),
            Trajectory(steps=[TrajectoryStep("q", "a2", 0.0, -0.3)]),
        ]
        data = compute_advantages(trajectories, MockTokenizer())
        assert len(data) == 2
        assert all(isinstance(d, Datum) for d in data)


# ---------------------------------------------------------------------------
# Preference dataset
# ---------------------------------------------------------------------------


class TestPreferenceDataset:
    def test_len(self):
        from cookbook.preference import Comparison, PreferenceDataset

        comps = [Comparison("prompt", "good", "bad")]
        ds = PreferenceDataset(comps, MockTokenizer())
        assert len(ds) == 1

    def test_iter(self):
        from cookbook.preference import Comparison, PreferenceDataset

        comps = [Comparison("prompt", "good", "bad")]
        ds = PreferenceDataset(comps, MockTokenizer())
        pairs = list(ds)
        assert len(pairs) == 1
        chosen, rejected = pairs[0]
        assert isinstance(chosen, Datum)
        assert isinstance(rejected, Datum)

    def test_getitem(self):
        from cookbook.preference import Comparison, PreferenceDataset

        comps = [Comparison("prompt", "good", "bad")]
        ds = PreferenceDataset(comps, MockTokenizer())
        chosen, rejected = ds[0]
        assert isinstance(chosen, Datum)

    def test_batch(self):
        from cookbook.preference import Comparison, PreferenceDataset

        comps = [Comparison(f"p{i}", f"g{i}", f"b{i}") for i in range(5)]
        ds = PreferenceDataset(comps, MockTokenizer())
        batches = list(ds.batch(2))
        assert len(batches) == 3

    def test_from_jsonl(self):
        from cookbook.preference import PreferenceDataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"prompt": "p", "chosen": "g", "rejected": "b"}) + "\n")
            path = f.name

        try:
            ds = PreferenceDataset.from_jsonl(path, MockTokenizer())
            assert len(ds) == 1
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Completers
# ---------------------------------------------------------------------------


class TestTokenCompleter:
    def test_complete(self):
        from cookbook.completers import TokenCompleter
        from local_tinker.sampling_client import SamplingClient
        from local_tinker.types import SamplingParams

        # Use a mock sampling client
        mock_sc = type("MockSC", (), {
            "sample": lambda self, prompt, params: SampleResponse(
                tokens=[1, 2], text="hello", log_probs=[-0.1, -0.2]
            )
        })()

        completer = TokenCompleter(mock_sc)
        result = completer.complete([10, 20, 30])
        assert result.text == "hello"


class TestMessageCompleter:
    def test_complete(self):
        from cookbook.completers import MessageCompleter
        from local_tinker.types import SamplingParams

        mock_sc = type("MockSC", (), {
            "sample": lambda self, prompt, params: SampleResponse(
                tokens=[1], text="response", log_probs=[-0.1]
            )
        })()

        completer = MessageCompleter(mock_sc, MockTokenizer())
        result = completer.complete([{"role": "user", "content": "hi"}])
        assert result == "response"
