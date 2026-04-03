"""Training metrics logging abstraction."""

from __future__ import annotations

import csv
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class MetricsLogger(ABC):
    """Abstract metrics logger interface."""

    @abstractmethod
    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics at a given step."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Finalize and flush any pending writes."""
        ...


class CSVLogger(MetricsLogger):
    """Log metrics to a CSV file. Always available, no extra dependencies.

    Args:
        path: Path to the CSV output file.
    """

    def __init__(self, path: str = "./training_log.csv") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._writer: csv.DictWriter | None = None
        self._file = None
        self._fieldnames: list[str] = []

    def log(self, metrics: dict[str, float], step: int) -> None:
        row = {"step": step, **metrics}

        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._file = open(self._path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()

        # Handle new fields appearing mid-training
        for key in row:
            if key not in self._fieldnames:
                self._fieldnames.append(key)
                # Reopen with new fieldnames
                if self._file:
                    self._file.close()
                self._file = open(self._path, "a", newline="")
                self._writer = csv.DictWriter(
                    self._file, fieldnames=self._fieldnames, extrasaction="ignore"
                )

        self._writer.writerow(row)
        if self._file:
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None


class WandbLogger(MetricsLogger):
    """Log metrics to Weights & Biases.

    Args:
        project: W&B project name.
        run_name: Optional run name.
        config: Optional config dict to log.
    """

    def __init__(
        self,
        project: str = "local-tinker",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        import wandb

        self._wandb = wandb
        wandb.init(project=project, name=run_name, config=config)

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._wandb.log(metrics, step=step)

    def close(self) -> None:
        self._wandb.finish()


class TensorBoardLogger(MetricsLogger):
    """Log metrics to TensorBoard.

    Args:
        log_dir: Directory for TensorBoard logs.
    """

    def __init__(self, log_dir: str = "./tb_logs") -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=log_dir)

    def log(self, metrics: dict[str, float], step: int) -> None:
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, step)

    def close(self) -> None:
        self._writer.close()


def set_logger(
    backend: str = "csv",
    **kwargs: Any,
) -> MetricsLogger:
    """Create a metrics logger for the specified backend.

    Args:
        backend: One of ``"csv"``, ``"wandb"``, ``"tensorboard"``.
        **kwargs: Backend-specific arguments.

    Returns:
        A :class:`MetricsLogger` instance.

    Raises:
        ValueError: If the backend is not recognized.
    """
    if backend == "csv":
        return CSVLogger(**kwargs)
    elif backend == "wandb":
        return WandbLogger(**kwargs)
    elif backend == "tensorboard":
        return TensorBoardLogger(**kwargs)
    else:
        raise ValueError(
            f"Unknown logging backend '{backend}'. "
            f"Supported: csv, wandb, tensorboard"
        )
