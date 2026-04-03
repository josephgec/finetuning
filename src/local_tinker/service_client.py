"""ServiceClient — entrypoint for creating training runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .types import LoraConfig

if TYPE_CHECKING:
    from .sampling_client import SamplingClient
    from .training_client import TrainingClient


class TrainingRun:
    """A live training run wrapping a loaded model + tokenizer.

    Created by ``ServiceClient.create_training_run``.  Provides access to
    :class:`TrainingClient` and :class:`SamplingClient`, which share the
    same underlying model instance.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        model_name: str,
        device: torch.device,
        lora_config: LoraConfig,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._device = device
        self._lora_config = lora_config

    # -- public helpers ------------------------------------------------

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def lora_config(self) -> LoraConfig:
        return self._lora_config

    # -- client factories ----------------------------------------------

    def training_client(self) -> TrainingClient:
        """Return a :class:`TrainingClient` bound to this run's model."""
        from .training_client import TrainingClient

        return TrainingClient(
            model=self._model,
            tokenizer=self._tokenizer,
            device=self._device,
        )

    def sampling_client(self) -> SamplingClient:
        """Return a :class:`SamplingClient` bound to this run's model."""
        from .sampling_client import SamplingClient

        return SamplingClient(
            model=self._model,
            tokenizer=self._tokenizer,
            device=self._device,
        )


class ServiceClient:
    """Top-level client that creates training runs.

    Args:
        device: PyTorch device string (``"cuda"``, ``"cpu"``, ``"mps"``,
            or ``"auto"`` to auto-detect).
    """

    def __init__(self, device: str = "auto") -> None:
        self._device = self._resolve_device(device)

    # -- public API ----------------------------------------------------

    def create_training_run(
        self,
        model: str,
        lora_config: LoraConfig | None = None,
        quantize: bool = False,
    ) -> TrainingRun:
        """Load a model with LoRA and return a :class:`TrainingRun`.

        Args:
            model: HuggingFace model identifier (e.g.
                ``"meta-llama/Llama-3.2-1B-Instruct"``).
            lora_config: LoRA adapter configuration. Uses sensible defaults
                if not provided.
            quantize: If ``True``, load in 4-bit via QLoRA (bitsandbytes).

        Returns:
            A :class:`TrainingRun` wrapping the loaded model.
        """
        if lora_config is None:
            lora_config = LoraConfig()

        # --- Load base model ------------------------------------------
        load_kwargs: dict = {
            "trust_remote_code": True,
        }

        if quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = {"": self._device}

        base_model = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)

        # --- Attach LoRA adapter --------------------------------------
        peft_config = PeftLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.rank,
            lora_alpha=int(lora_config.alpha),
            target_modules=lora_config.target_modules,
            lora_dropout=lora_config.dropout,
        )
        peft_model = get_peft_model(base_model, peft_config)

        # --- Tokenizer ------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return TrainingRun(
            model=peft_model,
            tokenizer=tokenizer,
            model_name=model,
            device=self._device,
            lora_config=lora_config,
        )

    # -- internal helpers ----------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)
