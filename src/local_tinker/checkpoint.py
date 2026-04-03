"""Checkpoint save / load / list for LoRA training runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .types import CheckpointMeta, LoraConfig


def save_checkpoint(
    training_client: Any,
    path: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a full checkpoint: LoRA weights, optimizer state, and metadata.

    Args:
        training_client: A :class:`TrainingClient` instance.
        path: Directory to write the checkpoint into.
        metadata: Optional user metadata (e.g. ``{"eval_loss": 0.42}``).

    Returns:
        Path to the checkpoint directory.
    """
    ckpt_dir = Path(path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save LoRA adapter weights
    training_client._model.save_pretrained(str(ckpt_dir))

    # 2. Save optimizer state (if optimizer exists)
    if training_client._optimizer is not None:
        torch.save(training_client._optimizer.state_dict(), ckpt_dir / "optimizer.pt")

    # 3. Save metadata
    meta = CheckpointMeta(
        model_name=getattr(training_client, "_model_name", "unknown"),
        step=training_client.get_step(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        lora_config=LoraConfig(),  # placeholder — caller can override via metadata
        metadata=metadata or {},
    )
    (ckpt_dir / "meta.json").write_text(meta.model_dump_json(indent=2))

    return ckpt_dir


def load_checkpoint(training_client: Any, path: str) -> CheckpointMeta:
    """Restore a checkpoint: LoRA weights, optimizer state, and step count.

    Args:
        training_client: A :class:`TrainingClient` instance.
        path: Directory containing a previously saved checkpoint.

    Returns:
        The :class:`CheckpointMeta` that was stored with the checkpoint.
    """
    ckpt_dir = Path(path)

    # 1. Load metadata
    meta_path = ckpt_dir / "meta.json"
    meta = CheckpointMeta(**json.loads(meta_path.read_text()))

    # 2. Restore LoRA weights
    training_client.load_weights(str(ckpt_dir))

    # 3. Restore optimizer state (if present and optimizer exists)
    opt_path = ckpt_dir / "optimizer.pt"
    if opt_path.exists() and training_client._optimizer is not None:
        training_client._optimizer.load_state_dict(
            torch.load(opt_path, map_location=training_client._device, weights_only=True)
        )

    # 4. Restore step count
    training_client._step = meta.step

    return meta


def list_checkpoints(directory: str) -> list[CheckpointMeta]:
    """Scan a directory for checkpoint subdirectories and return their metadata.

    Args:
        directory: Parent directory containing checkpoint folders.

    Returns:
        List of :class:`CheckpointMeta` sorted by step number.
    """
    root = Path(directory)
    if not root.is_dir():
        return []

    checkpoints: list[CheckpointMeta] = []
    for child in sorted(root.iterdir()):
        meta_path = child / "meta.json" if child.is_dir() else None
        if meta_path and meta_path.exists():
            meta = CheckpointMeta(**json.loads(meta_path.read_text()))
            checkpoints.append(meta)

    checkpoints.sort(key=lambda m: m.step)
    return checkpoints
