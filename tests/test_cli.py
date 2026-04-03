"""Tests for CLI commands."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from local_tinker.cli.main import app

runner = CliRunner()


class TestModelsCommand:
    def test_lists_models(self):
        result = runner.invoke(app, ["models"])
        assert result.exit_code == 0
        assert "Llama" in result.output
        assert "VRAM" in result.output


class TestInfoCommand:
    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_no_gpu(self, *_):
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "No GPU" in result.output or "CPU" in result.output.upper() or "No GPU" in result.output

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_mps(self, *_):
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "Apple" in result.output or "MPS" in result.output


class TestCheckpointListCommand:
    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, ["checkpoint", "list", tmpdir])
            assert result.exit_code == 0
            assert "No checkpoints" in result.output

    def test_with_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake checkpoint
            ckpt = Path(tmpdir) / "step-5"
            ckpt.mkdir()
            meta = {
                "model_name": "test-model",
                "step": 5,
                "timestamp": "2025-01-01T00:00:00",
                "lora_config": {"rank": 16, "alpha": 32.0, "target_modules": ["q_proj"], "dropout": 0.05},
                "metadata": {"loss": 0.5},
            }
            (ckpt / "meta.json").write_text(json.dumps(meta))

            result = runner.invoke(app, ["checkpoint", "list", tmpdir])
            assert result.exit_code == 0
            assert "test-model" in result.output
            assert "5" in result.output


class TestCheckpointInspectCommand:
    def test_inspect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = {
                "model_name": "test-model",
                "step": 10,
                "timestamp": "2025-01-01T00:00:00",
                "lora_config": {"rank": 16, "alpha": 32.0, "target_modules": ["q_proj"], "dropout": 0.05},
                "metadata": {},
            }
            (Path(tmpdir) / "meta.json").write_text(json.dumps(meta))

            result = runner.invoke(app, ["checkpoint", "inspect", tmpdir])
            assert result.exit_code == 0
            assert "test-model" in result.output

    def test_inspect_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, ["checkpoint", "inspect", tmpdir])
            assert result.exit_code == 1


class TestCheckpointExportCommand:
    def test_lora_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "ckpt"
            src.mkdir()
            (src / "adapter_model.safetensors").write_bytes(b"fake")
            (src / "adapter_config.json").write_text("{}")
            (src / "optimizer.pt").write_bytes(b"skip")

            dst = Path(tmpdir) / "out"
            result = runner.invoke(app, [
                "checkpoint", "export", str(src), "--output", str(dst), "--format", "lora"
            ])
            assert result.exit_code == 0
            assert (dst / "adapter_model.safetensors").exists()
            assert (dst / "adapter_config.json").exists()
            assert not (dst / "optimizer.pt").exists()

    def test_unknown_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, [
                "checkpoint", "export", tmpdir, "--format", "invalid"
            ])
            assert result.exit_code == 1
