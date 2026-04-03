# CLAUDE.md — Local Tinker

## What This Project Is
A Python library that provides Tinker-style fine-tuning primitives for local GPU training of small LLMs (1B–13B) using LoRA.

## Core Principles
1. **Tinker-compatible API surface**: The user-facing API should mirror Tinker's design — ServiceClient creates runs, TrainingClient handles forward_backward/optim_step, SamplingClient handles generation.
2. **Single-GPU simplicity**: No distributed training. Everything runs on one device. The value is the clean abstraction, not the infra.
3. **LoRA-only**: We only support LoRA/QLoRA fine-tuning, not full-parameter training.
4. **Synchronous-first, async-optional**: Unlike cloud Tinker (which needs async for network round-trips), our local version should default to synchronous calls. Provide async wrappers for users who want them.

## Key Architecture Decisions
- Models are loaded once via ServiceClient and held in GPU memory for the lifetime of the training run.
- TrainingClient and SamplingClient share the same underlying model instance — no need for weight syncing.
- forward_backward accumulates gradients on the LoRA parameters. optim_step applies them. This matches Tinker's two-phase design.
- Loss functions are objects passed to forward_backward, not arbitrary Python closures. They receive model logits and return a scalar loss.

## Code Style
- Python 3.10+, type hints everywhere
- Pydantic v2 for all config/type objects
- No global state — everything flows through client objects
- Docstrings on all public methods (Google style)
- Tests use pytest, mock GPU calls where possible

## Common Commands
- `uv run pytest` — run tests
- `uv run python examples/hello_tinker.py` — smoke test
- `uv pip install -e ".[dev]"` — install in dev mode
