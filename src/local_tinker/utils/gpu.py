"""GPU memory tracking and device selection utilities."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class GPUInfo:
    """GPU device information."""

    name: str
    total_vram_gb: float
    free_vram_gb: float
    used_vram_gb: float
    device_index: int


def get_gpu_info(device_index: int = 0) -> GPUInfo | None:
    """Return GPU info for the given device index.

    Args:
        device_index: CUDA device index (default 0).

    Returns:
        :class:`GPUInfo` or ``None`` if no GPU is available.
    """
    if not torch.cuda.is_available():
        return None

    props = torch.cuda.get_device_properties(device_index)
    total = props.total_mem / (1024 ** 3)
    allocated = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
    free = total - allocated

    return GPUInfo(
        name=props.name,
        total_vram_gb=round(total, 2),
        free_vram_gb=round(free, 2),
        used_vram_gb=round(allocated, 2),
        device_index=device_index,
    )


def get_all_gpus() -> list[GPUInfo]:
    """Return info for all available GPUs."""
    if not torch.cuda.is_available():
        return []
    return [
        info
        for i in range(torch.cuda.device_count())
        if (info := get_gpu_info(i)) is not None
    ]


def estimate_memory(
    params_billions: float,
    lora_rank: int = 16,
    quantize: bool = False,
) -> dict[str, float]:
    """Estimate VRAM requirements for a model configuration.

    Args:
        params_billions: Model size in billions of parameters.
        lora_rank: LoRA rank.
        quantize: Whether 4-bit quantization is used.

    Returns:
        Dict with ``model_gb``, ``lora_gb``, ``optimizer_gb``, ``total_gb``.
    """
    if quantize:
        bytes_per_param = 0.5  # 4-bit
    else:
        bytes_per_param = 2.0  # bfloat16

    model_gb = params_billions * bytes_per_param
    # LoRA params are roughly (rank * hidden * 2 * num_layers * num_targets)
    # Simplified estimate: ~0.5-2% of model params
    lora_fraction = lora_rank / 1000.0
    lora_gb = model_gb * lora_fraction
    # Optimizer states: ~2x the LoRA params (Adam momentum + variance)
    optimizer_gb = lora_gb * 2
    # Activations / overhead: ~20% of model
    overhead_gb = model_gb * 0.2

    total = model_gb + lora_gb + optimizer_gb + overhead_gb

    return {
        "model_gb": round(model_gb, 2),
        "lora_gb": round(lora_gb, 2),
        "optimizer_gb": round(optimizer_gb, 2),
        "total_gb": round(total, 2),
    }


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""

    allocated_gb: float
    reserved_gb: float
    peak_allocated_gb: float


@contextmanager
def track_memory(device_index: int = 0) -> Iterator[MemorySnapshot]:
    """Context manager that tracks GPU memory usage.

    Usage::

        with track_memory() as snapshot:
            # do work...
        print(f"Peak: {snapshot.peak_allocated_gb:.2f} GB")

    Args:
        device_index: CUDA device index.

    Yields:
        :class:`MemorySnapshot` updated on exit.
    """
    snapshot = MemorySnapshot(
        allocated_gb=0.0,
        reserved_gb=0.0,
        peak_allocated_gb=0.0,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_index)

    try:
        yield snapshot
    finally:
        if torch.cuda.is_available():
            snapshot.allocated_gb = round(
                torch.cuda.memory_allocated(device_index) / (1024 ** 3), 2
            )
            snapshot.reserved_gb = round(
                torch.cuda.memory_reserved(device_index) / (1024 ** 3), 2
            )
            snapshot.peak_allocated_gb = round(
                torch.cuda.max_memory_allocated(device_index) / (1024 ** 3), 2
            )
