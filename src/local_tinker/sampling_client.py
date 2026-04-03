"""SamplingClient — text generation / inference primitives."""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

from .types import ModelInput, SampleResponse, SamplingParams


class SamplingClient:
    """Wraps a model for inference and text generation.

    Created via :meth:`TrainingRun.sampling_client`.  Shares the same
    underlying model as the :class:`TrainingClient` — no weight syncing
    is needed.

    Args:
        model: PEFT-wrapped (or plain) causal LM.
        tokenizer: Associated tokenizer.
        device: Device the model lives on.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    # -- public API ----------------------------------------------------

    def sample(
        self, prompt: ModelInput, params: SamplingParams | None = None
    ) -> SampleResponse:
        """Generate a completion for a single prompt.

        Args:
            prompt: Tokenized prompt (see :class:`ModelInput`).
            params: Generation hyper-parameters. Uses defaults if ``None``.

        Returns:
            :class:`SampleResponse` with generated tokens, text, and
            optionally per-token log-probabilities.
        """
        if params is None:
            params = SamplingParams()

        self._model.eval()
        input_ids = prompt.ids.unsqueeze(0).to(self._device)

        with torch.no_grad():
            output = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=params.max_tokens,
                temperature=params.temperature if params.temperature > 0 else 1.0,
                top_p=params.top_p,
                top_k=params.top_k,
                do_sample=params.temperature > 0,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Extract only the newly generated tokens (exclude prompt)
        prompt_len = input_ids.shape[1]
        generated_ids = output.sequences[0, prompt_len:].tolist()
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute per-token log-probabilities from scores
        log_probs = self._extract_log_probs(output.scores, generated_ids)

        return SampleResponse(
            tokens=generated_ids,
            text=generated_text,
            log_probs=log_probs,
        )

    def batch_sample(
        self, prompts: list[ModelInput], params: SamplingParams | None = None
    ) -> list[SampleResponse]:
        """Generate completions for multiple prompts.

        Currently processes prompts sequentially. Batch padding support
        may be added in a future release.

        Args:
            prompts: List of tokenized prompts.
            params: Shared generation hyper-parameters.

        Returns:
            List of :class:`SampleResponse`, one per prompt.
        """
        return [self.sample(p, params) for p in prompts]

    # -- internal helpers ----------------------------------------------

    @staticmethod
    def _extract_log_probs(
        scores: tuple[torch.Tensor, ...], generated_ids: list[int]
    ) -> list[float]:
        """Compute log-probs of the actually generated tokens from scores.

        Args:
            scores: Per-step logit tensors from ``model.generate``.
            generated_ids: The token IDs that were generated.

        Returns:
            List of log-probability floats, one per generated token.
        """
        log_probs: list[float] = []
        for step_idx, token_id in enumerate(generated_ids):
            if step_idx >= len(scores):
                break
            step_logits = scores[step_idx][0]  # (vocab_size,)
            step_log_probs = torch.log_softmax(step_logits, dim=-1)
            log_probs.append(step_log_probs[token_id].item())
        return log_probs
