"""Tests for Local Tinker type definitions."""

import torch

from local_tinker.types import (
    AdamParams,
    CheckpointMeta,
    Datum,
    ForwardBackwardOutput,
    LoraConfig,
    ModelInput,
    OptimStepResponse,
    SampleResponse,
    SamplingParams,
)


class TestLoraConfig:
    def test_defaults(self):
        cfg = LoraConfig()
        assert cfg.rank == 16
        assert cfg.alpha == 32.0
        assert cfg.target_modules == ["q_proj", "v_proj"]
        assert cfg.dropout == 0.05

    def test_custom(self):
        cfg = LoraConfig(rank=8, alpha=16.0, target_modules=["q_proj"], dropout=0.1)
        assert cfg.rank == 8
        assert cfg.target_modules == ["q_proj"]

    def test_serialize_roundtrip(self):
        cfg = LoraConfig(rank=32, alpha=64.0)
        data = cfg.model_dump()
        restored = LoraConfig(**data)
        assert restored == cfg


class TestSamplingParams:
    def test_defaults(self):
        p = SamplingParams()
        assert p.max_tokens == 256
        assert p.temperature == 0.7
        assert p.stop == []

    def test_serialize_roundtrip(self):
        p = SamplingParams(max_tokens=128, temperature=0.0, stop=["###"])
        restored = SamplingParams(**p.model_dump())
        assert restored == p


class TestAdamParams:
    def test_defaults(self):
        p = AdamParams()
        assert p.lr == 2e-4
        assert p.betas == (0.9, 0.999)

    def test_custom(self):
        p = AdamParams(lr=1e-5, weight_decay=0.01)
        assert p.lr == 1e-5
        assert p.weight_decay == 0.01


class TestForwardBackwardOutput:
    def test_basic(self):
        out = ForwardBackwardOutput(loss=2.5, num_tokens=100, grad_norm=1.23)
        assert out.loss == 2.5
        assert out.num_tokens == 100
        assert out.grad_norm == 1.23

    def test_no_grad_norm(self):
        out = ForwardBackwardOutput(loss=1.0, num_tokens=50)
        assert out.grad_norm is None


class TestOptimStepResponse:
    def test_basic(self):
        r = OptimStepResponse(step=5, lr=1e-4)
        assert r.step == 5
        assert r.lr == 1e-4


class TestSampleResponse:
    def test_basic(self):
        r = SampleResponse(tokens=[1, 2, 3], text="hello", log_probs=[-0.5, -1.0, -0.3])
        assert len(r.tokens) == 3
        assert r.text == "hello"
        assert len(r.log_probs) == 3

    def test_no_log_probs(self):
        r = SampleResponse(tokens=[1], text="hi")
        assert r.log_probs is None


class TestDatum:
    def test_minimal(self):
        d = Datum(input_ids=[1, 2, 3])
        assert d.labels is None
        assert d.attention_mask is None

    def test_full(self):
        d = Datum(input_ids=[1, 2, 3], labels=[1, 2, 3], attention_mask=[1, 1, 1])
        assert d.labels == [1, 2, 3]


class TestModelInput:
    def test_from_ids(self):
        mi = ModelInput.from_ids([10, 20, 30])
        assert mi.ids.tolist() == [10, 20, 30]
        assert mi.ids.dtype == torch.long

    def test_from_text(self):
        # Use a mock tokenizer
        class MockTokenizer:
            def __call__(self, text, return_tensors=None):
                class Result:
                    input_ids = torch.tensor([[1, 2, 3]])
                return Result()

        mi = ModelInput.from_text("test", MockTokenizer())
        assert mi.ids.tolist() == [1, 2, 3]


class TestCheckpointMeta:
    def test_serialize(self):
        meta = CheckpointMeta(
            model_name="test-model",
            step=100,
            timestamp="2025-01-01T00:00:00",
            lora_config=LoraConfig(),
            metadata={"eval_loss": 0.42},
        )
        data = meta.model_dump()
        restored = CheckpointMeta(**data)
        assert restored.step == 100
        assert restored.metadata["eval_loss"] == 0.42
