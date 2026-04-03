"""TrainingClient — forward_backward / optim_step training loop primitives."""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

from .losses.base import LossFunction
from .types import AdamParams, Datum, ForwardBackwardOutput, OptimStepResponse


class TrainingClient:
    """Wraps a PEFT model for LoRA training.

    Created via :meth:`TrainingRun.training_client`.  Provides a two-phase
    training loop: ``forward_backward`` accumulates gradients, and
    ``optim_step`` applies them.

    Args:
        model: PEFT-wrapped causal LM.
        tokenizer: Associated tokenizer.
        device: Device the model lives on.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        auto_checkpoint_every: int | None = None,
        checkpoint_dir: str = "./checkpoints",
        max_checkpoints: int = 3,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._optimizer: torch.optim.AdamW | None = None
        self._current_lr: float | None = None
        self._step: int = 0
        self._auto_checkpoint_every = auto_checkpoint_every
        self._checkpoint_dir = checkpoint_dir
        self._max_checkpoints = max_checkpoints

    # -- public API ----------------------------------------------------

    def forward_backward(
        self, data: list[Datum], loss_fn: LossFunction
    ) -> ForwardBackwardOutput:
        """Run a forward + backward pass, accumulating gradients.

        Gradients are **not** applied — call :meth:`optim_step` to step.

        Args:
            data: Batch of training examples.
            loss_fn: Loss function to compute the objective.

        Returns:
            :class:`ForwardBackwardOutput` with loss value and token count.
        """
        self._model.train()

        input_ids, labels, attention_mask = self._collate(data)

        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = loss_fn.compute(outputs.logits, labels)
        loss.backward()

        # Count non-masked tokens
        num_tokens = int((labels != -100).sum().item())

        # Compute gradient norm (LoRA params only)
        grad_norm = self._grad_norm()

        return ForwardBackwardOutput(
            loss=loss.item(),
            num_tokens=num_tokens,
            grad_norm=grad_norm,
        )

    def optim_step(self, params: AdamParams) -> OptimStepResponse:
        """Apply accumulated gradients with Adam and zero them.

        The optimizer is created lazily on the first call. If ``params.lr``
        changes between calls the learning rate is updated in-place.

        Args:
            params: Adam hyper-parameters for this step.

        Returns:
            :class:`OptimStepResponse` with the new step count and LR.
        """
        self._ensure_optimizer(params)
        self._optimizer.step()  # type: ignore[union-attr]
        self._optimizer.zero_grad()  # type: ignore[union-attr]
        self._step += 1

        # Auto-checkpoint
        if (
            self._auto_checkpoint_every
            and self._step % self._auto_checkpoint_every == 0
        ):
            self._auto_checkpoint()

        return OptimStepResponse(step=self._step, lr=params.lr)

    def get_step(self) -> int:
        """Return the current training step count."""
        return self._step

    def save_weights(self, path: str) -> None:
        """Save the LoRA adapter weights to *path*."""
        self._model.save_pretrained(path)

    def load_weights(self, path: str) -> None:
        """Load LoRA adapter weights from *path*."""
        from peft import PeftModel

        if isinstance(self._model, PeftModel):
            self._model.load_adapter(path, adapter_name="default")
        else:
            raise TypeError("Model is not a PeftModel; cannot load adapter weights.")

    def get_reference_log_probs(self, data: list[Datum]) -> torch.Tensor:
        """Compute per-token log-probs under the base model (LoRA disabled).

        Useful for DPO and PPO KL penalties that need reference log-probs.

        Args:
            data: Batch of examples.

        Returns:
            (batch, seq_len) tensor of log-probabilities from the base model.
        """
        from peft import PeftModel

        input_ids, labels, attention_mask = self._collate(data)

        self._model.eval()
        with torch.no_grad():
            if isinstance(self._model, PeftModel):
                with self._model.disable_adapter():
                    outputs = self._model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
            else:
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

        from .utils.logprobs import get_per_token_logprobs

        return get_per_token_logprobs(outputs.logits, labels)

    # -- internal helpers ----------------------------------------------

    def _collate(
        self, data: list[Datum]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad and batch a list of Datum into tensors on device."""
        pad_id = self._tokenizer.pad_token_id or 0

        max_len = max(len(d.input_ids) for d in data)

        batch_input_ids: list[list[int]] = []
        batch_labels: list[list[int]] = []
        batch_attn: list[list[int]] = []

        for d in data:
            seq_len = len(d.input_ids)
            pad_len = max_len - seq_len

            batch_input_ids.append(d.input_ids + [pad_id] * pad_len)

            if d.labels is not None:
                batch_labels.append(d.labels + [-100] * pad_len)
            else:
                # Default: labels = input_ids (standard CLM)
                batch_labels.append(d.input_ids + [-100] * pad_len)

            if d.attention_mask is not None:
                batch_attn.append(d.attention_mask + [0] * pad_len)
            else:
                batch_attn.append([1] * seq_len + [0] * pad_len)

        input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=self._device)
        labels = torch.tensor(batch_labels, dtype=torch.long, device=self._device)
        attention_mask = torch.tensor(batch_attn, dtype=torch.long, device=self._device)

        return input_ids, labels, attention_mask

    def _ensure_optimizer(self, params: AdamParams) -> None:
        """Create or update the optimizer to match *params*."""
        if self._optimizer is None or self._current_lr != params.lr:
            trainable = [p for p in self._model.parameters() if p.requires_grad]
            self._optimizer = torch.optim.AdamW(
                trainable,
                lr=params.lr,
                betas=params.betas,
                weight_decay=params.weight_decay,
                eps=params.eps,
            )
            self._current_lr = params.lr

    def _auto_checkpoint(self) -> None:
        """Save a checkpoint and rotate old ones."""
        import shutil
        from pathlib import Path

        from .checkpoint import save_checkpoint

        ckpt_path = Path(self._checkpoint_dir) / f"step-{self._step}"
        save_checkpoint(self, str(ckpt_path))

        # Rotate: keep only the most recent max_checkpoints
        root = Path(self._checkpoint_dir)
        if root.is_dir():
            ckpt_dirs = sorted(
                [d for d in root.iterdir() if d.is_dir() and (d / "meta.json").exists()],
                key=lambda d: d.stat().st_mtime,
            )
            while len(ckpt_dirs) > self._max_checkpoints:
                shutil.rmtree(ckpt_dirs.pop(0))

    def _grad_norm(self) -> float | None:
        """Compute the L2 norm of gradients on trainable parameters."""
        grads = [
            p.grad for p in self._model.parameters() if p.requires_grad and p.grad is not None
        ]
        if not grads:
            return None
        total_norm = torch.norm(
            torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2
        )
        return total_norm.item()
