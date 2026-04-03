"""Tests for model registry."""

from local_tinker.model_registry import (
    get_default_lora_config,
    get_model_info,
    list_models,
    recommend_models,
)
from local_tinker.types import LoraConfig

import pytest


class TestListModels:
    def test_returns_list(self):
        models = list_models()
        assert len(models) > 0

    def test_sorted_by_params(self):
        models = list_models()
        params = [m.params_billions for m in models]
        assert params == sorted(params)


class TestGetModelInfo:
    def test_known_model(self):
        info = get_model_info("meta-llama/Llama-3.2-1B-Instruct")
        assert info is not None
        assert info.params_billions == 1.2
        assert info.architecture == "dense"

    def test_unknown_model(self):
        info = get_model_info("nonexistent/model")
        assert info is None

    def test_vram_estimates_reasonable(self):
        info = get_model_info("meta-llama/Llama-3.1-8B-Instruct")
        assert info is not None
        assert info.vram_4bit_gb < info.vram_fp16_gb
        assert info.vram_4bit_gb > 0


class TestGetDefaultLoraConfig:
    def test_returns_lora_config(self):
        config = get_default_lora_config("meta-llama/Llama-3.2-1B-Instruct")
        assert isinstance(config, LoraConfig)
        assert "q_proj" in config.target_modules

    def test_8b_model_has_more_targets(self):
        config = get_default_lora_config("meta-llama/Llama-3.1-8B-Instruct")
        assert len(config.target_modules) == 4

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            get_default_lora_config("nonexistent/model")


class TestRecommendModels:
    def test_large_budget(self):
        recs = recommend_models(24.0)
        assert len(recs) > 0
        # All should fit
        for m in recs:
            assert m.vram_4bit_gb <= 24.0

    def test_small_budget(self):
        recs = recommend_models(3.0)
        assert len(recs) >= 1  # At least the 1B model

    def test_tiny_budget(self):
        recs = recommend_models(0.5)
        assert len(recs) == 0

    def test_sorted_largest_first(self):
        recs = recommend_models(20.0)
        params = [m.params_billions for m in recs]
        assert params == sorted(params, reverse=True)
