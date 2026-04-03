"""Tests for ServiceClient and TrainingRun using mocks (no GPU/model downloads)."""

from unittest.mock import MagicMock, patch

import torch

from local_tinker.service_client import ServiceClient, TrainingRun
from local_tinker.training_client import TrainingClient
from local_tinker.sampling_client import SamplingClient
from local_tinker.types import LoraConfig


# ---------------------------------------------------------------------------
# TrainingRun tests (no mocking needed — just test the wrapper)
# ---------------------------------------------------------------------------

class TestTrainingRun:
    def _make_run(self):
        model = MagicMock()
        tokenizer = MagicMock()
        lora_config = LoraConfig()
        return TrainingRun(
            model=model,
            tokenizer=tokenizer,
            model_name="test/model",
            device=torch.device("cpu"),
            lora_config=lora_config,
        )

    def test_properties(self):
        run = self._make_run()
        assert run.model_name == "test/model"
        assert run.device == torch.device("cpu")
        assert isinstance(run.lora_config, LoraConfig)
        assert run.tokenizer is not None

    def test_training_client_returns_correct_type(self):
        run = self._make_run()
        tc = run.training_client()
        assert isinstance(tc, TrainingClient)

    def test_sampling_client_returns_correct_type(self):
        run = self._make_run()
        sc = run.sampling_client()
        assert isinstance(sc, SamplingClient)

    def test_clients_share_model(self):
        run = self._make_run()
        tc = run.training_client()
        sc = run.sampling_client()
        assert tc._model is sc._model


# ---------------------------------------------------------------------------
# ServiceClient._resolve_device tests
# ---------------------------------------------------------------------------

class TestResolveDevice:
    def test_explicit_cpu(self):
        device = ServiceClient._resolve_device("cpu")
        assert device == torch.device("cpu")

    def test_explicit_cuda(self):
        device = ServiceClient._resolve_device("cuda")
        assert device == torch.device("cuda")

    def test_explicit_mps(self):
        device = ServiceClient._resolve_device("mps")
        assert device == torch.device("mps")

    @patch("torch.cuda.is_available", return_value=True)
    def test_auto_selects_cuda(self, mock_cuda):
        device = ServiceClient._resolve_device("auto")
        assert device == torch.device("cuda")

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_auto_selects_mps(self, mock_mps, mock_cuda):
        device = ServiceClient._resolve_device("auto")
        assert device == torch.device("mps")

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_auto_falls_back_to_cpu(self, mock_mps, mock_cuda):
        device = ServiceClient._resolve_device("auto")
        assert device == torch.device("cpu")


# ---------------------------------------------------------------------------
# ServiceClient.create_training_run tests (mocked model loading)
# ---------------------------------------------------------------------------

class TestCreateTrainingRun:
    @patch("local_tinker.service_client.get_peft_model")
    @patch("local_tinker.service_client.AutoTokenizer")
    @patch("local_tinker.service_client.AutoModelForCausalLM")
    def test_basic_creation(self, MockModel, MockTokenizer, mock_get_peft):
        mock_base = MagicMock()
        MockModel.from_pretrained.return_value = mock_base

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "PAD"
        MockTokenizer.from_pretrained.return_value = mock_tokenizer

        mock_peft = MagicMock()
        mock_get_peft.return_value = mock_peft

        client = ServiceClient(device="cpu")
        lora = LoraConfig(rank=8, alpha=16.0)
        run = client.create_training_run(model="test/model", lora_config=lora)

        assert isinstance(run, TrainingRun)
        assert run.model_name == "test/model"
        MockModel.from_pretrained.assert_called_once()
        MockTokenizer.from_pretrained.assert_called_once()
        mock_get_peft.assert_called_once()

    @patch("local_tinker.service_client.get_peft_model")
    @patch("local_tinker.service_client.AutoTokenizer")
    @patch("local_tinker.service_client.AutoModelForCausalLM")
    def test_default_lora_config(self, MockModel, MockTokenizer, mock_get_peft):
        MockModel.from_pretrained.return_value = MagicMock()
        mock_tok = MagicMock()
        mock_tok.pad_token = "PAD"
        MockTokenizer.from_pretrained.return_value = mock_tok
        mock_get_peft.return_value = MagicMock()

        client = ServiceClient(device="cpu")
        run = client.create_training_run(model="test/model")

        # Default LoraConfig should be used
        assert run.lora_config == LoraConfig()

    @patch("local_tinker.service_client.get_peft_model")
    @patch("local_tinker.service_client.AutoTokenizer")
    @patch("local_tinker.service_client.AutoModelForCausalLM")
    def test_pad_token_set_when_missing(self, MockModel, MockTokenizer, mock_get_peft):
        MockModel.from_pretrained.return_value = MagicMock()
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"
        MockTokenizer.from_pretrained.return_value = mock_tok
        mock_get_peft.return_value = MagicMock()

        client = ServiceClient(device="cpu")
        run = client.create_training_run(model="test/model")

        assert mock_tok.pad_token == "<eos>"

    @patch("local_tinker.service_client.BitsAndBytesConfig")
    @patch("local_tinker.service_client.get_peft_model")
    @patch("local_tinker.service_client.AutoTokenizer")
    @patch("local_tinker.service_client.AutoModelForCausalLM")
    def test_quantize_uses_bnb_config(self, MockModel, MockTokenizer, mock_get_peft, MockBnB):
        MockModel.from_pretrained.return_value = MagicMock()
        mock_tok = MagicMock()
        mock_tok.pad_token = "PAD"
        MockTokenizer.from_pretrained.return_value = mock_tok
        mock_get_peft.return_value = MagicMock()
        MockBnB.return_value = "bnb_config_obj"

        client = ServiceClient(device="cpu")
        run = client.create_training_run(model="test/model", quantize=True)

        MockBnB.assert_called_once()
        call_kwargs = MockModel.from_pretrained.call_args[1]
        assert call_kwargs["quantization_config"] == "bnb_config_obj"
        assert call_kwargs["device_map"] == "auto"

    @patch("local_tinker.service_client.get_peft_model")
    @patch("local_tinker.service_client.AutoTokenizer")
    @patch("local_tinker.service_client.AutoModelForCausalLM")
    def test_no_quantize_uses_bfloat16(self, MockModel, MockTokenizer, mock_get_peft):
        MockModel.from_pretrained.return_value = MagicMock()
        mock_tok = MagicMock()
        mock_tok.pad_token = "PAD"
        MockTokenizer.from_pretrained.return_value = mock_tok
        mock_get_peft.return_value = MagicMock()

        client = ServiceClient(device="cpu")
        run = client.create_training_run(model="test/model", quantize=False)

        call_kwargs = MockModel.from_pretrained.call_args[1]
        assert call_kwargs["torch_dtype"] == torch.bfloat16
        assert "quantization_config" not in call_kwargs
