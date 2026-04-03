"""All shared types for Local Tinker."""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, Field


class LoraConfig(BaseModel):
    """Configuration for LoRA adapter layers."""

    rank: int = 16
    alpha: float = 32.0
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    dropout: float = 0.05


class SamplingParams(BaseModel):
    """Parameters for text generation / sampling."""

    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop: list[str] = Field(default_factory=list)


class AdamParams(BaseModel):
    """Parameters for the Adam optimizer step."""

    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    eps: float = 1e-8


class ForwardBackwardOutput(BaseModel):
    """Result of a forward_backward call."""

    loss: float
    num_tokens: int
    grad_norm: float | None = None

    model_config = {"arbitrary_types_allowed": True}


class OptimStepResponse(BaseModel):
    """Result of an optim_step call."""

    step: int
    lr: float


class SampleResponse(BaseModel):
    """Result of a sample call."""

    tokens: list[int]
    text: str
    log_probs: list[float] | None = None


class Datum(BaseModel):
    """A single training example with token IDs and optional labels/mask."""

    input_ids: list[int]
    labels: list[int] | None = None
    attention_mask: list[int] | None = None

    model_config = {"arbitrary_types_allowed": True}


class ModelInput:
    """Input for sampling — wraps token IDs as a tensor."""

    def __init__(self, ids: torch.Tensor) -> None:
        self.ids = ids

    @classmethod
    def from_text(cls, text: str, tokenizer: Any) -> ModelInput:
        """Create ModelInput by tokenizing a text string."""
        encoded = tokenizer(text, return_tensors="pt")
        return cls(ids=encoded.input_ids[0])

    @classmethod
    def from_ids(cls, token_ids: list[int]) -> ModelInput:
        """Create ModelInput from a list of token IDs."""
        return cls(ids=torch.tensor(token_ids, dtype=torch.long))


class CheckpointMeta(BaseModel):
    """Metadata stored alongside a checkpoint."""

    model_name: str
    step: int
    timestamp: str
    lora_config: LoraConfig
    metadata: dict[str, Any] = Field(default_factory=dict)
