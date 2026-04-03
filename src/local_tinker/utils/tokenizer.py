"""Tokenizer helpers and chat template rendering utilities."""

from __future__ import annotations

from typing import Any


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> list[int]:
    """Apply the tokenizer's built-in chat template, or fall back to a simple format.

    Args:
        tokenizer: HuggingFace tokenizer.
        messages: List of ``{"role": str, "content": str}`` dicts.
        add_generation_prompt: Whether to add a generation prompt at the end.

    Returns:
        List of token IDs.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )

    # Fallback: simple text format
    parts = []
    for msg in messages:
        parts.append(f"{msg['role']}: {msg['content']}")
    if add_generation_prompt:
        parts.append("assistant:")
    text = "\n".join(parts)
    return tokenizer(text).input_ids


def ensure_pad_token(tokenizer: Any) -> None:
    """Set ``pad_token`` to ``eos_token`` if not already set.

    Args:
        tokenizer: HuggingFace tokenizer to modify in-place.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def count_tokens(tokenizer: Any, text: str) -> int:
    """Count the number of tokens in a text string.

    Args:
        tokenizer: HuggingFace tokenizer.
        text: Input text.

    Returns:
        Number of tokens.
    """
    return len(tokenizer(text).input_ids)
