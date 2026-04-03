# Local Tinker

**Tinker-style API for local LoRA fine-tuning of small language models (1B–13B) on a single GPU.**

Local Tinker gives you clean, high-level primitives — `forward_backward`, `optim_step`, `sample` — for fine-tuning open-weight LLMs on your own hardware. It handles model loading, LoRA management, gradient accumulation, and inference internally so you can focus on writing training loops.

Inspired by [Thinking Machines' Tinker API](https://tinker-docs.thinkingmachines.ai/), but everything runs locally on a single machine instead of a managed cloud cluster.

---

## Architecture

```mermaid
graph TB
    subgraph User Code
        Script["Training Script"]
    end

    subgraph Local Tinker API
        SC["ServiceClient"]
        TR["TrainingRun"]
        TC["TrainingClient"]
        SaC["SamplingClient"]
        LF["Loss Functions"]
    end

    subgraph Backend
        HF["HuggingFace Transformers"]
        PEFT["PEFT / LoRA"]
        BNB["bitsandbytes / QLoRA"]
        GPU["Local GPU"]
    end

    Script --> SC
    SC -->|create_training_run| TR
    TR -->|training_client| TC
    TR -->|sampling_client| SaC
    TC -->|forward_backward| LF
    TC -->|optim_step| GPU
    SaC -->|sample| GPU
    SC --> HF
    SC --> PEFT
    SC --> BNB
    HF --> GPU
    PEFT --> GPU
```

## Training Loop Flow

```mermaid
sequenceDiagram
    participant U as User Script
    participant SC as ServiceClient
    participant TR as TrainingRun
    participant TC as TrainingClient
    participant Sa as SamplingClient
    participant M as Model on GPU

    U->>SC: create ServiceClient
    U->>SC: create_training_run
    SC->>M: Load model + LoRA adapter
    SC-->>U: TrainingRun

    U->>TR: training_client
    TR-->>U: TrainingClient
    U->>TR: sampling_client
    TR-->>U: SamplingClient

    loop Training Steps
        U->>TC: forward_backward
        TC->>M: train, forward, loss, backward
        TC-->>U: ForwardBackwardOutput

        U->>TC: optim_step
        TC->>M: optimizer step + zero grad
        TC-->>U: OptimStepResponse
    end

    U->>Sa: sample
    Sa->>M: eval + generate
    Sa-->>U: SampleResponse
```

## Gradient Accumulation (Two-Phase Design)

```mermaid
graph LR
    subgraph "Phase 1: Accumulate Gradients"
        FB1["forward_backward<br/>micro-batch 1"] --> G["Gradients<br/>accumulate"]
        FB2["forward_backward<br/>micro-batch 2"] --> G
        FB3["forward_backward<br/>micro-batch 3"] --> G
        FB4["forward_backward<br/>micro-batch 4"] --> G
    end

    subgraph "Phase 2: Apply"
        G --> OS["optim_step<br/>Adam optimizer"]
        OS --> ZG["zero_grad"]
    end
```

## Loss Function Hierarchy

```mermaid
classDiagram
    class LossFunction {
        <<abstract>>
        +compute&#40;logits, labels&#41; Tensor
    }
    class CrossEntropyLoss {
        +mask_prompt_tokens bool
        +compute&#40;logits, labels&#41; Tensor
    }
    class PPOLoss {
        +clip_range float
        +kl_coeff float
    }
    class GRPOLoss {
        +clip_range float
        +kl_coeff float
    }
    class DPOLoss {
        +beta float
    }
    class CustomLoss {
        +fn Callable
    }

    LossFunction <|-- CrossEntropyLoss
    LossFunction <|-- PPOLoss
    LossFunction <|-- GRPOLoss
    LossFunction <|-- DPOLoss
    LossFunction <|-- CustomLoss
```

## RL Training Loop

```mermaid
graph LR
    subgraph "1. Rollout"
        E["Environment<br/>ProblemEnv"] -->|observation| SC["SamplingClient<br/>sample"]
        SC -->|action| E
        E -->|reward| T["Trajectory"]
    end

    subgraph "2. Advantage"
        T --> CA["compute_advantages<br/>GRPO"]
        CA --> D["Training Data"]
    end

    subgraph "3. Policy Update"
        D --> FB["TrainingClient<br/>forward_backward"]
        FB --> OS["TrainingClient<br/>optim_step"]
    end
```

---

## Installation

```bash
# Clone and install
git clone https://github.com/josephgec/finetuning.git
cd finetuning/local-tinker

# Using uv (recommended)
uv venv && uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- PyTorch 2.1+
- A CUDA-compatible GPU (or MPS on Apple Silicon, or CPU for testing)

---

## Quick Start

```python
from local_tinker import (
    ServiceClient, LoraConfig, SamplingParams, AdamParams,
    ModelInput, Datum,
)
from local_tinker.losses import CrossEntropyLoss

# 1. Create a training run
client = ServiceClient()  # auto-detects GPU
run = client.create_training_run(
    model="meta-llama/Llama-3.2-1B-Instruct",
    lora_config=LoraConfig(rank=16, alpha=32, target_modules=["q_proj", "v_proj"]),
)
tc = run.training_client()
sc = run.sampling_client()

# 2. Train (one SFT step)
text = "The capital of France is Paris."
encoded = run.tokenizer(text, return_tensors="pt")
ids = encoded.input_ids[0].tolist()

result = tc.forward_backward(
    data=[Datum(input_ids=ids, labels=ids)],
    loss_fn=CrossEntropyLoss(),
)
print(f"Loss: {result.loss:.4f}")
tc.optim_step(AdamParams(lr=1e-4))

# 3. Sample
output = sc.sample(
    prompt=ModelInput.from_text("The capital of France is", run.tokenizer),
    params=SamplingParams(max_tokens=20, temperature=0.7),
)
print(f"Generated: {output.text}")
```

---

## Core Concepts

### ServiceClient

The entrypoint. Creates training runs by loading a model with LoRA adapters.

```python
client = ServiceClient(device="auto")  # "cuda", "mps", "cpu", or "auto"

run = client.create_training_run(
    model="meta-llama/Llama-3.2-3B-Instruct",
    lora_config=LoraConfig(rank=16, alpha=32),
    quantize=True,  # 4-bit QLoRA — fits larger models on smaller GPUs
)
```

### TrainingClient

Wraps the model for training. Uses a two-phase design: `forward_backward` accumulates gradients, `optim_step` applies them.

```python
tc = run.training_client()

# Gradient accumulation: call forward_backward multiple times
for micro_batch in split(batch, chunks=4):
    tc.forward_backward(micro_batch, loss_fn=CrossEntropyLoss())

# Then apply all accumulated gradients at once
tc.optim_step(AdamParams(lr=2e-4))
```

| Method | Description |
|---|---|
| `forward_backward(data, loss_fn)` | Forward + backward pass. Accumulates gradients. |
| `optim_step(params)` | Applies gradients with Adam, then zeros them. |
| `get_step()` | Returns current training step count. |
| `save_weights(path)` | Saves LoRA adapter weights to disk. |
| `load_weights(path)` | Loads LoRA adapter weights from disk. |

### SamplingClient

Wraps the model for inference. Shares the same model instance as TrainingClient — no weight syncing needed.

```python
sc = run.sampling_client()

response = sc.sample(
    prompt=ModelInput.from_text("Explain quantum computing:", tokenizer),
    params=SamplingParams(max_tokens=256, temperature=0.7, top_p=0.9),
)
print(response.text)
print(response.log_probs)  # per-token log probabilities
```

| Method | Description |
|---|---|
| `sample(prompt, params)` | Generate a single completion. |
| `batch_sample(prompts, params)` | Generate completions for multiple prompts. |

### Loss Functions

Loss functions are objects passed to `forward_backward`. They receive model logits and return a scalar loss.

```python
from local_tinker.losses import CrossEntropyLoss

loss_fn = CrossEntropyLoss(mask_prompt_tokens=True)
result = tc.forward_backward(data, loss_fn)
```

| Loss | Description |
|---|---|
| `CrossEntropyLoss` | Standard next-token prediction (SFT). Supports prompt masking via `-100` labels. |
| `DPOLoss` | Direct Preference Optimization — chosen vs rejected pairs with reference model. |
| `PPOLoss` | PPO clipped surrogate objective with optional KL penalty. |
| `GRPOLoss` | Group Relative Policy Optimization — normalizes rewards within groups. |
| `CustomLoss` | Wrap any `callable(logits, labels) -> scalar` as a loss function. |

---

## API Reference

### Types

All types use Pydantic v2 `BaseModel` for validation and serialization.

```python
from local_tinker import (
    LoraConfig,       # rank, alpha, target_modules, dropout
    SamplingParams,   # max_tokens, temperature, top_p, top_k, stop
    AdamParams,       # lr, betas, weight_decay, eps
    Datum,            # input_ids, labels, attention_mask
    ModelInput,       # .from_text(str, tokenizer) / .from_ids(list[int])
    ForwardBackwardOutput,  # loss, num_tokens, grad_norm
    OptimStepResponse,      # step, lr
    SampleResponse,         # tokens, text, log_probs
)
```

### LoraConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `rank` | `int` | `16` | LoRA rank (r) |
| `alpha` | `float` | `32.0` | LoRA alpha scaling |
| `target_modules` | `list[str]` | `["q_proj", "v_proj"]` | Modules to attach LoRA to |
| `dropout` | `float` | `0.05` | LoRA dropout rate |

### SamplingParams

| Field | Type | Default | Description |
|---|---|---|---|
| `max_tokens` | `int` | `256` | Maximum tokens to generate |
| `temperature` | `float` | `0.7` | Sampling temperature (0 = greedy) |
| `top_p` | `float` | `0.9` | Nucleus sampling threshold |
| `top_k` | `int` | `50` | Top-k sampling |
| `stop` | `list[str]` | `[]` | Stop sequences |

### AdamParams

| Field | Type | Default | Description |
|---|---|---|---|
| `lr` | `float` | `2e-4` | Learning rate |
| `betas` | `tuple[float, float]` | `(0.9, 0.999)` | Adam beta parameters |
| `weight_decay` | `float` | `0.0` | Weight decay |
| `eps` | `float` | `1e-8` | Adam epsilon |

---

## 4-bit QLoRA

Fit larger models on smaller GPUs by enabling 4-bit quantization:

```python
run = client.create_training_run(
    model="meta-llama/Llama-3.1-8B-Instruct",
    lora_config=LoraConfig(rank=16, alpha=32),
    quantize=True,  # 4-bit QLoRA via bitsandbytes
)
```

This uses NF4 quantization with double quantization and bfloat16 compute dtype for stability.

---

## Model Compatibility

| Model | Params | Min VRAM (4-bit) | Min VRAM (fp16) | Default LoRA Targets |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 1.2B | ~3 GB | ~5 GB | q_proj, v_proj |
| Llama-3.2-3B-Instruct | 3.2B | ~4 GB | ~8 GB | q_proj, v_proj |
| Qwen2.5-3B-Instruct | 3B | ~4 GB | ~8 GB | q_proj, v_proj |
| Phi-3.5-mini-instruct | 3.8B | ~5 GB | ~9 GB | q_proj, v_proj |
| Llama-3.1-8B-Instruct | 8B | ~6 GB | ~17 GB | q_proj, v_proj, k_proj, o_proj |
| Mistral-7B-Instruct-v0.3 | 7.2B | ~6 GB | ~16 GB | q_proj, v_proj |
| Qwen2.5-7B-Instruct | 7B | ~6 GB | ~16 GB | q_proj, v_proj |
| Gemma-2-9B-it | 9.2B | ~7 GB | ~20 GB | q_proj, v_proj |

Any HuggingFace causal LM model works — the table above lists tested configurations.

---

## Project Structure

```
local-tinker/
├── pyproject.toml                     # Package configuration
├── CLAUDE.md                          # Claude Code instructions
├── src/local_tinker/
│   ├── __init__.py                    # Public API exports
│   ├── types.py                       # Pydantic types & config objects
│   ├── config.py                      # (reserved)
│   ├── model_registry.py              # Supported model catalog + VRAM estimates
│   ├── service_client.py              # ServiceClient + TrainingRun
│   ├── training_client.py             # TrainingClient (forward_backward, optim_step)
│   ├── sampling_client.py             # SamplingClient (sample, batch_sample)
│   ├── checkpoint.py                  # Save/load/list LoRA checkpoints
│   ├── weights.py                     # Merge LoRA → full model, export, push to Hub
│   ├── losses/
│   │   ├── base.py                    # Abstract LossFunction interface
│   │   ├── cross_entropy.py           # SFT cross-entropy loss
│   │   ├── dpo.py                     # Direct Preference Optimization
│   │   ├── ppo.py                     # PPO clipped surrogate loss
│   │   ├── grpo.py                    # Group Relative Policy Optimization
│   │   └── custom.py                  # User-defined scalar loss
│   ├── utils/
│   │   ├── gpu.py                     # GPU memory tracking, device selection
│   │   ├── tokenizer.py              # Tokenizer helpers, chat templates
│   │   ├── logging.py                # CSV / W&B / TensorBoard logging
│   │   └── logprobs.py               # Per-token log-probability extraction
│   └── cli/
│       └── main.py                    # CLI entrypoint (run, models, info, checkpoint)
├── cookbook/
│   ├── supervised.py                  # SupervisedDataset, ChatDatasetBuilder
│   ├── rl.py                          # Env, MessageEnv, ProblemEnv, rollout
│   ├── preference.py                  # Comparison, PreferenceDataset
│   ├── renderers.py                   # Llama3/Qwen/Mistral chat renderers
│   ├── completers.py                  # TokenCompleter, MessageCompleter
│   └── recipes/
│       ├── chat_sft.py                # End-to-end chat SFT
│       ├── math_rl.py                 # Math RL with verifier rewards
│       ├── code_rl.py                 # Code RL with execution rewards
│       ├── dpo_preference.py          # DPO preference tuning
│       └── distillation.py            # Knowledge distillation
├── examples/
│   ├── hello_tinker.py                # Minimal end-to-end example
│   ├── sft_alpaca.py                  # SFT on Alpaca dataset
│   ├── rl_gsm8k.py                    # RL on GSM8K math problems
│   └── dpo_ultrafeedback.py           # DPO on preference data
└── tests/                             # 179 tests, high coverage
    ├── test_types.py
    ├── test_losses.py
    ├── test_training_client.py
    ├── test_sampling_client.py
    ├── test_service_client.py
    ├── test_checkpoint.py
    ├── test_weights.py
    ├── test_model_registry.py
    ├── test_utils.py
    ├── test_cookbook.py
    └── test_cli.py
```

---

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=local_tinker --cov-report=term-missing

# Run smoke test (requires GPU + model access)
uv run python examples/hello_tinker.py
```

---

## Tech Stack

| Component | Library | Purpose |
|---|---|---|
| Model loading | `transformers` | Load HuggingFace models |
| LoRA adapters | `peft` | Attach, train, save LoRA adapters |
| Quantization | `bitsandbytes` | 4-bit / 8-bit QLoRA |
| Tensor ops | `torch` | Autograd, optimizer, GPU memory |
| Config | `pydantic` v2 | Typed, validated config objects |
| CLI | `typer` | CLI commands (run, models, checkpoint) |
| Packaging | `pyproject.toml` + `uv` | Distribution |

---

## CLI

```bash
# List supported models with VRAM requirements
local-tinker models

# Show GPU info and recommended models
local-tinker info

# Run a training script
local-tinker run examples/hello_tinker.py

# Checkpoint management
local-tinker checkpoint list ./checkpoints
local-tinker checkpoint inspect ./checkpoints/step-100
local-tinker checkpoint export ./checkpoints/step-100 --format lora
```

---

## Checkpoint & Weight Management

```python
from local_tinker.checkpoint import save_checkpoint, load_checkpoint, list_checkpoints
from local_tinker.weights import merge_and_save, export_lora, push_to_hub

# Save checkpoint (LoRA weights + optimizer + metadata)
save_checkpoint(tc, "./checkpoints/step-100", metadata={"eval_loss": 0.42})

# Load checkpoint
meta = load_checkpoint(tc, "./checkpoints/step-100")

# List all checkpoints
for ckpt in list_checkpoints("./checkpoints"):
    print(f"Step {ckpt.step}: {ckpt.metadata}")

# Auto-checkpointing (on TrainingClient)
tc = run.training_client()
tc._auto_checkpoint_every = 50  # checkpoint every 50 steps
tc._checkpoint_dir = "./checkpoints"
tc._max_checkpoints = 3  # keep only latest 3

# Merge LoRA into base model and save
merge_and_save(tc, "./merged-model")

# Export just the lightweight LoRA adapter
export_lora(tc, "./lora-adapter")

# Push merged model to HuggingFace Hub
push_to_hub(tc, "my-username/my-model", private=True)
```

---

## Cookbook

The `cookbook/` module provides high-level abstractions for common tasks:

```python
# Supervised fine-tuning
from cookbook.supervised import SupervisedDataset, ChatDatasetBuilder
dataset = SupervisedDataset.from_jsonl("data.jsonl", tokenizer)
for batch in dataset.batch(4):
    tc.forward_backward(batch, CrossEntropyLoss())
    tc.optim_step(AdamParams(lr=2e-4))

# RL training
from cookbook.rl import ProblemEnv, rollout, compute_advantages
env = ProblemEnv(problems, extract_answer_fn=extract_number)
trajectories = [rollout(env, sc, tokenizer) for _ in range(8)]
data = compute_advantages(trajectories, tokenizer, method="grpo")

# DPO preference tuning
from cookbook.preference import PreferenceDataset
dataset = PreferenceDataset.from_jsonl("prefs.jsonl", tokenizer)

# Chat completion
from cookbook.completers import MessageCompleter
completer = MessageCompleter(sc, tokenizer)
response = completer.complete([{"role": "user", "content": "Hello!"}])
```

### Ready-to-run Recipes

| Recipe | Script | Description |
|---|---|---|
| Chat SFT | `cookbook/recipes/chat_sft.py` | Fine-tune on chat conversations |
| Math RL | `cookbook/recipes/math_rl.py` | RL on math problems with verifier |
| Code RL | `cookbook/recipes/code_rl.py` | RL on code with execution rewards |
| DPO | `cookbook/recipes/dpo_preference.py` | DPO preference tuning |
| Distillation | `cookbook/recipes/distillation.py` | KL distillation from larger model |

---

## Design Principles

1. **Tinker-compatible API surface** — `ServiceClient` creates runs, `TrainingClient` handles training, `SamplingClient` handles generation.
2. **Single-GPU simplicity** — No distributed training. Everything runs on one device.
3. **LoRA-only** — Only LoRA/QLoRA fine-tuning, not full-parameter training.
4. **Two-phase training** — `forward_backward` accumulates gradients, `optim_step` applies them. This enables gradient accumulation with zero extra code.
5. **Shared model instance** — Training and sampling clients share the same model. No weight syncing needed.

---

## Roadmap

- [x] **Phase 1**: Core primitives (ServiceClient, TrainingClient, SamplingClient, CrossEntropyLoss)
- [x] **Phase 2**: RL loss functions (PPO, GRPO, DPO, CustomLoss) + reference model support
- [x] **Phase 3**: Checkpoint management + weight export/merge/push to Hub
- [x] **Phase 4**: Cookbook (datasets, RL environments, renderers, completers, 5 recipes)
- [x] **Phase 5**: CLI (run, models, info, checkpoint) + GPU utilities + model registry + logging
- [x] **Phase 6**: 179 tests across 11 test modules

---

## License

MIT
