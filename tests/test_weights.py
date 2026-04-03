"""Tests for weights export module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from local_tinker.weights import export_lora, merge_and_save, push_to_hub


def _make_peft_mock():
    """Create a mock model that passes isinstance(_, PeftModel) check."""
    from peft import PeftModel

    model = MagicMock()
    # Patch the class check that weights.py does internally
    model.__class__ = PeftModel
    merged = MagicMock()
    model.merge_and_unload.return_value = merged
    return model, merged


class MockTrainingClient:
    def __init__(self):
        self._model = MagicMock()
        self._tokenizer = MagicMock()
        self._device = "cpu"


class TestMergeAndSave:
    def test_merges_peft_model(self):
        tc = MockTrainingClient()
        model, merged = _make_peft_mock()
        tc._model = model

        with tempfile.TemporaryDirectory() as tmpdir:
            result = merge_and_save(tc, tmpdir)
            model.merge_and_unload.assert_called_once()
            merged.save_pretrained.assert_called_once()
            tc._tokenizer.save_pretrained.assert_called_once()
            assert result == Path(tmpdir)

    def test_saves_non_peft_model(self):
        tc = MockTrainingClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            merge_and_save(tc, tmpdir)
            tc._model.save_pretrained.assert_called_once()


class TestExportLora:
    def test_calls_save_pretrained(self):
        tc = MockTrainingClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_lora(tc, tmpdir)
            tc._model.save_pretrained.assert_called_once()
            assert result == Path(tmpdir)

    def test_creates_directory(self):
        tc = MockTrainingClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = Path(tmpdir) / "nested" / "lora"
            export_lora(tc, str(new_path))
            assert new_path.exists()


class TestPushToHub:
    def test_pushes_merged_model(self):
        tc = MockTrainingClient()
        model, merged = _make_peft_mock()
        tc._model = model

        url = push_to_hub(tc, "user/my-model", private=True)
        model.merge_and_unload.assert_called_once()
        merged.push_to_hub.assert_called_once_with("user/my-model", private=True)
        tc._tokenizer.push_to_hub.assert_called_once()
        assert url == "https://huggingface.co/user/my-model"
