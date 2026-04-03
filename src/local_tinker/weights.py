"""Post-training weight management: merge, export, push to Hub."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def merge_and_save(training_client: Any, output_path: str) -> Path:
    """Merge LoRA weights into the base model and save the full model.

    Args:
        training_client: A :class:`TrainingClient` instance.
        output_path: Directory to write the merged model into.

    Returns:
        Path to the merged model directory.
    """
    from peft import PeftModel

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    model = training_client._model
    if isinstance(model, PeftModel):
        merged = model.merge_and_unload()
    else:
        merged = model

    merged.save_pretrained(str(out))

    # Also save the tokenizer if available
    if hasattr(training_client, "_tokenizer"):
        training_client._tokenizer.save_pretrained(str(out))

    return out


def export_lora(training_client: Any, output_path: str) -> Path:
    """Save only the LoRA adapter files (lightweight export).

    Args:
        training_client: A :class:`TrainingClient` instance.
        output_path: Directory to write adapter files into.

    Returns:
        Path to the adapter directory.
    """
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    training_client._model.save_pretrained(str(out))
    return out


def push_to_hub(
    training_client: Any,
    repo_id: str,
    private: bool = True,
) -> str:
    """Merge LoRA into the base model and push to HuggingFace Hub.

    Args:
        training_client: A :class:`TrainingClient` instance.
        repo_id: HuggingFace Hub repository ID (e.g. ``"user/my-model"``).
        private: Whether to create a private repository.

    Returns:
        The Hub URL for the pushed model.
    """
    from peft import PeftModel

    model = training_client._model
    if isinstance(model, PeftModel):
        merged = model.merge_and_unload()
    else:
        merged = model

    merged.push_to_hub(repo_id, private=private)

    if hasattr(training_client, "_tokenizer"):
        training_client._tokenizer.push_to_hub(repo_id, private=private)

    return f"https://huggingface.co/{repo_id}"
