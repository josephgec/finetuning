"""Preference dataset utilities for DPO training."""

from __future__ import annotations

import json
from typing import Any, Iterator

from local_tinker.types import Datum


class Comparison:
    """A single preference comparison: chosen vs rejected response.

    Args:
        prompt: The input prompt.
        chosen: The preferred response.
        rejected: The dispreferred response.
    """

    def __init__(self, prompt: str, chosen: str, rejected: str) -> None:
        self.prompt = prompt
        self.chosen = chosen
        self.rejected = rejected


class PreferenceDataset:
    """Dataset of preference comparisons for DPO training.

    Args:
        comparisons: List of :class:`Comparison` objects.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
    """

    def __init__(
        self,
        comparisons: list[Comparison],
        tokenizer: Any,
        max_length: int = 2048,
    ) -> None:
        self._comparisons = comparisons
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._comparisons)

    def __iter__(self) -> Iterator[tuple[Datum, Datum]]:
        for comp in self._comparisons:
            yield self._tokenize_pair(comp)

    def __getitem__(self, idx: int) -> tuple[Datum, Datum]:
        return self._tokenize_pair(self._comparisons[idx])

    def batch(self, batch_size: int) -> Iterator[list[tuple[Datum, Datum]]]:
        """Yield batches of (chosen, rejected) Datum pairs.

        Args:
            batch_size: Number of comparison pairs per batch.

        Yields:
            Lists of ``(chosen_datum, rejected_datum)`` tuples.
        """
        current: list[tuple[Datum, Datum]] = []
        for pair in self:
            current.append(pair)
            if len(current) == batch_size:
                yield current
                current = []
        if current:
            yield current

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        tokenizer: Any,
        prompt_col: str = "prompt",
        chosen_col: str = "chosen",
        rejected_col: str = "rejected",
        max_length: int = 2048,
    ) -> PreferenceDataset:
        """Load from a JSONL file.

        Args:
            path: Path to the JSONL file.
            tokenizer: HuggingFace tokenizer.
            prompt_col: Column name for the prompt.
            chosen_col: Column name for the chosen response.
            rejected_col: Column name for the rejected response.
            max_length: Maximum sequence length.

        Returns:
            A :class:`PreferenceDataset`.
        """
        comparisons = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    comparisons.append(
                        Comparison(
                            prompt=row[prompt_col],
                            chosen=row[chosen_col],
                            rejected=row[rejected_col],
                        )
                    )
        return cls(comparisons, tokenizer, max_length)

    def _tokenize_pair(self, comp: Comparison) -> tuple[Datum, Datum]:
        """Tokenize a chosen/rejected pair with prompt masking."""
        prompt_enc = self._tokenizer(comp.prompt, add_special_tokens=True)
        prompt_len = len(prompt_enc.input_ids)

        chosen_enc = self._tokenizer(
            comp.prompt + comp.chosen, add_special_tokens=True
        )
        rejected_enc = self._tokenizer(
            comp.prompt + comp.rejected, add_special_tokens=True
        )

        chosen_ids = chosen_enc.input_ids[: self._max_length]
        rejected_ids = rejected_enc.input_ids[: self._max_length]

        # Mask prompt in labels
        chosen_labels = [-100] * min(prompt_len, len(chosen_ids)) + chosen_ids[prompt_len:]
        rejected_labels = [-100] * min(prompt_len, len(rejected_ids)) + rejected_ids[prompt_len:]

        chosen_datum = Datum(input_ids=chosen_ids, labels=chosen_labels[: len(chosen_ids)])
        rejected_datum = Datum(input_ids=rejected_ids, labels=rejected_labels[: len(rejected_ids)])

        return chosen_datum, rejected_datum
