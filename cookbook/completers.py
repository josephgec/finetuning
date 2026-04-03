"""High-level completer wrappers around SamplingClient."""

from __future__ import annotations

from typing import Any

from local_tinker.sampling_client import SamplingClient
from local_tinker.types import ModelInput, SampleResponse, SamplingParams


class TokenCompleter:
    """Thin wrapper around SamplingClient for token-level control.

    Args:
        sampling_client: A :class:`SamplingClient` instance.
        default_params: Default sampling parameters.
    """

    def __init__(
        self,
        sampling_client: SamplingClient,
        default_params: SamplingParams | None = None,
    ) -> None:
        self._client = sampling_client
        self._default_params = default_params or SamplingParams()

    def complete(
        self,
        token_ids: list[int],
        params: SamplingParams | None = None,
    ) -> SampleResponse:
        """Generate a completion from token IDs.

        Args:
            token_ids: Input token IDs.
            params: Override default sampling parameters.

        Returns:
            :class:`SampleResponse` with generated tokens and text.
        """
        prompt = ModelInput.from_ids(token_ids)
        return self._client.sample(prompt, params or self._default_params)


class MessageCompleter:
    """Chat-aware completer that handles chat template formatting.

    Args:
        sampling_client: A :class:`SamplingClient` instance.
        tokenizer: HuggingFace tokenizer with ``apply_chat_template``.
        default_params: Default sampling parameters.
    """

    def __init__(
        self,
        sampling_client: SamplingClient,
        tokenizer: Any,
        default_params: SamplingParams | None = None,
    ) -> None:
        self._client = sampling_client
        self._tokenizer = tokenizer
        self._default_params = default_params or SamplingParams()

    def complete(
        self,
        messages: list[dict[str, str]],
        params: SamplingParams | None = None,
    ) -> str:
        """Generate an assistant response given a conversation.

        Args:
            messages: List of ``{"role": str, "content": str}`` dicts.
            params: Override default sampling parameters.

        Returns:
            The generated assistant message as a string.
        """
        if hasattr(self._tokenizer, "apply_chat_template"):
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            text += "\nassistant:"
            input_ids = self._tokenizer(text).input_ids

        prompt = ModelInput.from_ids(input_ids)
        response = self._client.sample(prompt, params or self._default_params)
        return response.text
