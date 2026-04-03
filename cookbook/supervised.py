"""Supervised dataset utilities for SFT training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from local_tinker.types import Datum


class SupervisedDataset:
    """A dataset of (input, output) text pairs for SFT training.

    Args:
        examples: List of ``{"input": str, "output": str}`` dicts.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length (truncates if exceeded).
    """

    def __init__(
        self,
        examples: list[dict[str, str]],
        tokenizer: Any,
        max_length: int = 2048,
    ) -> None:
        self._examples = examples
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._examples)

    def __iter__(self) -> Iterator[Datum]:
        for ex in self._examples:
            yield self._tokenize(ex["input"], ex["output"])

    def __getitem__(self, idx: int) -> Datum:
        ex = self._examples[idx]
        return self._tokenize(ex["input"], ex["output"])

    def batch(self, batch_size: int) -> Iterator[list[Datum]]:
        """Yield batches of Datum objects.

        Args:
            batch_size: Number of examples per batch.

        Yields:
            Lists of :class:`Datum` of length up to ``batch_size``.
        """
        current: list[Datum] = []
        for datum in self:
            current.append(datum)
            if len(current) == batch_size:
                yield current
                current = []
        if current:
            yield current

    @classmethod
    def from_jsonl(cls, path: str, tokenizer: Any, max_length: int = 2048) -> SupervisedDataset:
        """Load from a JSONL file with ``input`` and ``output`` fields.

        Args:
            path: Path to the JSONL file.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.

        Returns:
            A :class:`SupervisedDataset`.
        """
        examples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return cls(examples, tokenizer, max_length)

    @classmethod
    def from_hf_dataset(
        cls,
        dataset_name: str,
        split: str = "train",
        input_col: str = "input",
        output_col: str = "output",
        tokenizer: Any = None,
        max_length: int = 2048,
    ) -> SupervisedDataset:
        """Load from a HuggingFace datasets dataset.

        Args:
            dataset_name: HuggingFace dataset identifier.
            split: Dataset split (e.g. ``"train"``).
            input_col: Column name for input text.
            output_col: Column name for output text.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.

        Returns:
            A :class:`SupervisedDataset`.
        """
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split=split)
        examples = [
            {"input": row[input_col], "output": row[output_col]}
            for row in ds
        ]
        return cls(examples, tokenizer, max_length)

    def _tokenize(self, input_text: str, output_text: str) -> Datum:
        """Tokenize an input/output pair into a Datum with label masking."""
        prompt_enc = self._tokenizer(input_text, add_special_tokens=True)
        full_enc = self._tokenizer(
            input_text + output_text, add_special_tokens=True
        )

        input_ids = full_enc.input_ids[: self._max_length]
        prompt_len = len(prompt_enc.input_ids)

        # Mask prompt tokens in labels (only train on output)
        labels = [-100] * min(prompt_len, len(input_ids)) + input_ids[prompt_len:]
        labels = labels[: self._max_length]

        return Datum(input_ids=input_ids, labels=labels)


class ChatDatasetBuilder:
    """Build supervised training data from multi-turn chat conversations.

    Args:
        tokenizer: HuggingFace tokenizer with ``apply_chat_template`` support.
        max_length: Maximum sequence length.
        train_on_assistant_only: If True, only compute loss on assistant turns.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 2048,
        train_on_assistant_only: bool = True,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._train_on_assistant_only = train_on_assistant_only

    def build(self, conversations: list[list[dict[str, str]]]) -> list[Datum]:
        """Convert a list of conversations to Datum objects.

        Args:
            conversations: List of conversations. Each conversation is a list
                of ``{"role": str, "content": str}`` message dicts.

        Returns:
            List of :class:`Datum` objects ready for ``forward_backward``.
        """
        return [self._process_conversation(conv) for conv in conversations]

    def _process_conversation(self, messages: list[dict[str, str]]) -> Datum:
        """Process a single conversation into a Datum."""
        if hasattr(self._tokenizer, "apply_chat_template"):
            input_ids = self._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
        else:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            input_ids = self._tokenizer(text).input_ids

        input_ids = input_ids[: self._max_length]

        if self._train_on_assistant_only:
            labels = self._mask_non_assistant(messages, input_ids)
        else:
            labels = list(input_ids)

        return Datum(input_ids=list(input_ids), labels=labels)

    def _mask_non_assistant(
        self, messages: list[dict[str, str]], full_ids: list[int]
    ) -> list[int]:
        """Create labels that only train on assistant turn tokens."""
        labels = [-100] * len(full_ids)

        # Tokenize incrementally to find assistant turn boundaries
        token_pos = 0
        for i, msg in enumerate(messages):
            # Tokenize up to and including this message
            prefix = messages[: i + 1]
            if hasattr(self._tokenizer, "apply_chat_template"):
                prefix_ids = self._tokenizer.apply_chat_template(
                    prefix, tokenize=True, add_generation_prompt=False
                )
            else:
                text = "\n".join(f"{m['role']}: {m['content']}" for m in prefix)
                prefix_ids = self._tokenizer(text).input_ids

            end_pos = min(len(prefix_ids), len(full_ids))

            if msg["role"] == "assistant":
                for j in range(token_pos, end_pos):
                    labels[j] = full_ids[j]

            token_pos = end_pos

        return labels
