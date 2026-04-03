"""Tests for checkpoint save/load/list."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import torch

from local_tinker.checkpoint import list_checkpoints, load_checkpoint, save_checkpoint
from local_tinker.types import CheckpointMeta


class MockTrainingClient:
    """Minimal mock of TrainingClient for checkpoint tests."""

    def __init__(self):
        self._model = MagicMock()
        self._optimizer = None
        self._step = 5
        self._device = torch.device("cpu")

    def get_step(self):
        return self._step

    def load_weights(self, path):
        pass  # no-op


class TestSaveCheckpoint:
    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt", "step-5")
            tc = MockTrainingClient()
            result = save_checkpoint(tc, path)
            assert result.exists()
            assert (result / "meta.json").exists()

    def test_meta_json_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "step-5")
            tc = MockTrainingClient()
            save_checkpoint(tc, path, metadata={"eval_loss": 0.42})

            meta = json.loads((Path(path) / "meta.json").read_text())
            assert meta["step"] == 5
            assert meta["metadata"]["eval_loss"] == 0.42
            assert "timestamp" in meta

    def test_calls_save_pretrained(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "step-5")
            tc = MockTrainingClient()
            save_checkpoint(tc, path)
            tc._model.save_pretrained.assert_called_once()

    def test_saves_optimizer_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "step-5")
            tc = MockTrainingClient()
            # Give it a real optimizer
            param = torch.nn.Parameter(torch.zeros(4))
            tc._optimizer = torch.optim.AdamW([param], lr=1e-3)
            save_checkpoint(tc, path)
            assert (Path(path) / "optimizer.pt").exists()

    def test_no_optimizer_state_when_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "step-5")
            tc = MockTrainingClient()
            tc._optimizer = None
            save_checkpoint(tc, path)
            assert not (Path(path) / "optimizer.pt").exists()


class TestLoadCheckpoint:
    def test_restores_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "step-10")
            tc = MockTrainingClient()
            tc._step = 10
            save_checkpoint(tc, path)

            # Reset step and reload
            tc._step = 0
            meta = load_checkpoint(tc, path)
            assert tc._step == 10
            assert meta.step == 10

    def test_returns_meta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "step-5")
            tc = MockTrainingClient()
            save_checkpoint(tc, path, metadata={"note": "test"})

            meta = load_checkpoint(tc, path)
            assert isinstance(meta, CheckpointMeta)
            assert meta.metadata["note"] == "test"


class TestListCheckpoints:
    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = list_checkpoints(tmpdir)
            assert result == []

    def test_nonexistent_directory(self):
        result = list_checkpoints("/nonexistent/path")
        assert result == []

    def test_finds_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tc = MockTrainingClient()

            tc._step = 5
            save_checkpoint(tc, os.path.join(tmpdir, "step-5"))
            tc._step = 10
            save_checkpoint(tc, os.path.join(tmpdir, "step-10"))
            tc._step = 15
            save_checkpoint(tc, os.path.join(tmpdir, "step-15"))

            result = list_checkpoints(tmpdir)
            assert len(result) == 3
            assert result[0].step == 5
            assert result[1].step == 10
            assert result[2].step == 15

    def test_sorted_by_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tc = MockTrainingClient()

            # Create in reverse order
            tc._step = 20
            save_checkpoint(tc, os.path.join(tmpdir, "z-step-20"))
            tc._step = 5
            save_checkpoint(tc, os.path.join(tmpdir, "a-step-5"))

            result = list_checkpoints(tmpdir)
            assert result[0].step == 5
            assert result[1].step == 20
