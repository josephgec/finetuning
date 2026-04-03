"""Chat template renderers for different model families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from local_tinker.types import Datum


class Renderer(ABC):
    """Abstract renderer that converts chat messages into tokenized Datum."""

    @abstractmethod
    def render(
        self,
        messages: list[dict[str, str]],
        tokenizer: Any,
        max_length: int = 2048,
    ) -> Datum:
        """Convert a list of messages to a training-ready Datum.

        Args:
            messages: List of ``{"role": str, "content": str}`` dicts.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.

        Returns:
            A :class:`Datum` with labels masked on non-assistant turns.
        """
        ...


class Llama3Renderer(Renderer):
    """Renderer for Llama 3.x models using ``<|start_header_id|>`` format."""

    def render(
        self,
        messages: list[dict[str, str]],
        tokenizer: Any,
        max_length: int = 2048,
    ) -> Datum:
        parts: list[str] = []
        for msg in messages:
            parts.append(
                f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
                f"{msg['content']}<|eot_id|>"
            )
        text = "<|begin_of_text|>" + "".join(parts)
        return self._tokenize_with_mask(text, messages, tokenizer, max_length)

    def _tokenize_with_mask(
        self,
        text: str,
        messages: list[dict[str, str]],
        tokenizer: Any,
        max_length: int,
    ) -> Datum:
        input_ids = tokenizer(text, add_special_tokens=False).input_ids[:max_length]
        labels = [-100] * len(input_ids)

        # Find assistant content boundaries
        for msg in messages:
            if msg["role"] == "assistant":
                content_tokens = tokenizer(msg["content"], add_special_tokens=False).input_ids
                # Search for content tokens in input_ids
                for start in range(len(input_ids) - len(content_tokens) + 1):
                    if input_ids[start : start + len(content_tokens)] == content_tokens:
                        for j in range(start, min(start + len(content_tokens), len(input_ids))):
                            labels[j] = input_ids[j]
                        break

        return Datum(input_ids=input_ids, labels=labels)


class QwenRenderer(Renderer):
    """Renderer for Qwen models using ``<|im_start|>`` / ``<|im_end|>`` format."""

    def render(
        self,
        messages: list[dict[str, str]],
        tokenizer: Any,
        max_length: int = 2048,
    ) -> Datum:
        parts: list[str] = []
        for msg in messages:
            parts.append(
                f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            )
        text = "".join(parts)
        input_ids = tokenizer(text, add_special_tokens=False).input_ids[:max_length]
        labels = [-100] * len(input_ids)

        for msg in messages:
            if msg["role"] == "assistant":
                content_tokens = tokenizer(msg["content"], add_special_tokens=False).input_ids
                for start in range(len(input_ids) - len(content_tokens) + 1):
                    if input_ids[start : start + len(content_tokens)] == content_tokens:
                        for j in range(start, min(start + len(content_tokens), len(input_ids))):
                            labels[j] = input_ids[j]
                        break

        return Datum(input_ids=input_ids, labels=labels)


class MistralRenderer(Renderer):
    """Renderer for Mistral models using ``[INST]`` / ``[/INST]`` format."""

    def render(
        self,
        messages: list[dict[str, str]],
        tokenizer: Any,
        max_length: int = 2048,
    ) -> Datum:
        parts: list[str] = []
        for msg in messages:
            if msg["role"] == "user":
                parts.append(f"[INST] {msg['content']} [/INST]")
            elif msg["role"] == "assistant":
                parts.append(f"{msg['content']}</s>")
            elif msg["role"] == "system":
                parts.append(f"[INST] {msg['content']}\n")
        text = "<s>" + "".join(parts)
        input_ids = tokenizer(text, add_special_tokens=False).input_ids[:max_length]
        labels = [-100] * len(input_ids)

        for msg in messages:
            if msg["role"] == "assistant":
                content_tokens = tokenizer(msg["content"], add_special_tokens=False).input_ids
                for start in range(len(input_ids) - len(content_tokens) + 1):
                    if input_ids[start : start + len(content_tokens)] == content_tokens:
                        for j in range(start, min(start + len(content_tokens), len(input_ids))):
                            labels[j] = input_ids[j]
                        break

        return Datum(input_ids=input_ids, labels=labels)


_RENDERER_MAP: dict[str, type[Renderer]] = {
    "llama": Llama3Renderer,
    "qwen": QwenRenderer,
    "mistral": MistralRenderer,
}


def get_renderer(model_name: str) -> Renderer:
    """Auto-detect and return the appropriate renderer for a model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        A :class:`Renderer` instance.

    Raises:
        ValueError: If no renderer matches the model name.
    """
    name_lower = model_name.lower()
    for key, cls in _RENDERER_MAP.items():
        if key in name_lower:
            return cls()
    raise ValueError(
        f"No renderer found for model '{model_name}'. "
        f"Known families: {list(_RENDERER_MAP.keys())}"
    )
