"""Supported model catalog and loading helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .types import LoraConfig


@dataclass
class ModelInfo:
    """Metadata for a supported model."""

    name: str
    params_billions: float
    architecture: str
    vram_fp16_gb: float
    vram_4bit_gb: float
    default_lora_targets: list[str]
    chat_template: str


_REGISTRY: dict[str, ModelInfo] = {
    "meta-llama/Llama-3.2-1B-Instruct": ModelInfo(
        name="meta-llama/Llama-3.2-1B-Instruct",
        params_billions=1.2,
        architecture="dense",
        vram_fp16_gb=5.0,
        vram_4bit_gb=3.0,
        default_lora_targets=["q_proj", "v_proj"],
        chat_template="llama3",
    ),
    "meta-llama/Llama-3.2-3B-Instruct": ModelInfo(
        name="meta-llama/Llama-3.2-3B-Instruct",
        params_billions=3.2,
        architecture="dense",
        vram_fp16_gb=8.0,
        vram_4bit_gb=4.0,
        default_lora_targets=["q_proj", "v_proj"],
        chat_template="llama3",
    ),
    "Qwen/Qwen2.5-3B-Instruct": ModelInfo(
        name="Qwen/Qwen2.5-3B-Instruct",
        params_billions=3.0,
        architecture="dense",
        vram_fp16_gb=8.0,
        vram_4bit_gb=4.0,
        default_lora_targets=["q_proj", "v_proj"],
        chat_template="qwen",
    ),
    "microsoft/Phi-3.5-mini-instruct": ModelInfo(
        name="microsoft/Phi-3.5-mini-instruct",
        params_billions=3.8,
        architecture="dense",
        vram_fp16_gb=9.0,
        vram_4bit_gb=5.0,
        default_lora_targets=["q_proj", "v_proj"],
        chat_template="phi",
    ),
    "meta-llama/Llama-3.1-8B-Instruct": ModelInfo(
        name="meta-llama/Llama-3.1-8B-Instruct",
        params_billions=8.0,
        architecture="dense",
        vram_fp16_gb=17.0,
        vram_4bit_gb=6.0,
        default_lora_targets=["q_proj", "v_proj", "k_proj", "o_proj"],
        chat_template="llama3",
    ),
    "mistralai/Mistral-7B-Instruct-v0.3": ModelInfo(
        name="mistralai/Mistral-7B-Instruct-v0.3",
        params_billions=7.2,
        architecture="dense",
        vram_fp16_gb=16.0,
        vram_4bit_gb=6.0,
        default_lora_targets=["q_proj", "v_proj"],
        chat_template="mistral",
    ),
    "Qwen/Qwen2.5-7B-Instruct": ModelInfo(
        name="Qwen/Qwen2.5-7B-Instruct",
        params_billions=7.0,
        architecture="dense",
        vram_fp16_gb=16.0,
        vram_4bit_gb=6.0,
        default_lora_targets=["q_proj", "v_proj"],
        chat_template="qwen",
    ),
    "google/gemma-2-9b-it": ModelInfo(
        name="google/gemma-2-9b-it",
        params_billions=9.2,
        architecture="dense",
        vram_fp16_gb=20.0,
        vram_4bit_gb=7.0,
        default_lora_targets=["q_proj", "v_proj"],
        chat_template="gemma",
    ),
}


def list_models() -> list[ModelInfo]:
    """Return all models in the registry, sorted by parameter count."""
    return sorted(_REGISTRY.values(), key=lambda m: m.params_billions)


def get_model_info(name: str) -> ModelInfo | None:
    """Look up model metadata by HuggingFace identifier.

    Args:
        name: HuggingFace model identifier.

    Returns:
        :class:`ModelInfo` or ``None`` if not in the registry.
    """
    return _REGISTRY.get(name)


def get_default_lora_config(name: str) -> LoraConfig:
    """Return a sensible default LoRA config for a registered model.

    Args:
        name: HuggingFace model identifier.

    Returns:
        :class:`LoraConfig` with recommended settings.

    Raises:
        KeyError: If the model is not in the registry.
    """
    info = _REGISTRY.get(name)
    if info is None:
        raise KeyError(
            f"Model '{name}' not in registry. "
            f"Known models: {list(_REGISTRY.keys())}"
        )
    return LoraConfig(
        rank=16,
        alpha=32.0,
        target_modules=info.default_lora_targets,
        dropout=0.05,
    )


def recommend_models(available_vram_gb: float) -> list[ModelInfo]:
    """Suggest models that fit in the given VRAM budget (4-bit).

    Args:
        available_vram_gb: Available GPU VRAM in GB.

    Returns:
        List of :class:`ModelInfo` sorted by size (largest first).
    """
    fits = [m for m in _REGISTRY.values() if m.vram_4bit_gb <= available_vram_gb]
    return sorted(fits, key=lambda m: m.params_billions, reverse=True)
