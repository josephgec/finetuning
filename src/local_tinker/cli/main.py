"""CLI entrypoint for local-tinker."""

from __future__ import annotations

import subprocess
import sys

import typer

app = typer.Typer(
    name="local-tinker",
    help="Local Tinker — LoRA fine-tuning CLI for small language models.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# local-tinker run
# ---------------------------------------------------------------------------


@app.command()
def run(
    script: str = typer.Argument(..., help="Path to a Python training script."),
) -> None:
    """Run a training script."""
    typer.echo(f"Running: {script}")
    result = subprocess.run([sys.executable, script])
    raise typer.Exit(code=result.returncode)


# ---------------------------------------------------------------------------
# local-tinker models
# ---------------------------------------------------------------------------


@app.command()
def models() -> None:
    """List supported models with VRAM requirements."""
    from ..model_registry import list_models

    typer.echo(
        f"{'Model':<45} {'Params':>7} {'FP16 VRAM':>10} {'4-bit VRAM':>11}"
    )
    typer.echo("-" * 75)
    for m in list_models():
        typer.echo(
            f"{m.name:<45} {m.params_billions:>6.1f}B "
            f"{m.vram_fp16_gb:>8.1f} GB {m.vram_4bit_gb:>9.1f} GB"
        )


# ---------------------------------------------------------------------------
# local-tinker info
# ---------------------------------------------------------------------------


@app.command()
def info() -> None:
    """Show GPU info and recommended models."""
    import torch

    from ..model_registry import recommend_models
    from ..utils.gpu import get_all_gpus

    if not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            typer.echo("Device: Apple Silicon (MPS)")
            typer.echo("Note: VRAM estimates are for CUDA GPUs.")
        else:
            typer.echo("No GPU detected. Training will run on CPU (slow).")
        return

    gpus = get_all_gpus()
    for gpu in gpus:
        typer.echo(f"GPU {gpu.device_index}: {gpu.name}")
        typer.echo(f"  Total VRAM: {gpu.total_vram_gb:.1f} GB")
        typer.echo(f"  Free VRAM:  {gpu.free_vram_gb:.1f} GB")
        typer.echo(f"  Used VRAM:  {gpu.used_vram_gb:.1f} GB")

        typer.echo("\n  Recommended models (4-bit):")
        recs = recommend_models(gpu.free_vram_gb)
        if recs:
            for m in recs[:5]:
                typer.echo(
                    f"    - {m.name} ({m.params_billions:.1f}B, "
                    f"~{m.vram_4bit_gb:.0f} GB)"
                )
        else:
            typer.echo("    (none — not enough free VRAM)")


# ---------------------------------------------------------------------------
# local-tinker checkpoint (subcommands)
# ---------------------------------------------------------------------------

checkpoint_app = typer.Typer(help="Manage training checkpoints.", no_args_is_help=True)
app.add_typer(checkpoint_app, name="checkpoint")


@checkpoint_app.command("list")
def checkpoint_list(
    directory: str = typer.Argument("./checkpoints", help="Directory with checkpoints."),
) -> None:
    """List checkpoints in a directory."""
    from ..checkpoint import list_checkpoints

    ckpts = list_checkpoints(directory)
    if not ckpts:
        typer.echo("No checkpoints found.")
        return

    typer.echo(f"{'Step':>8}  {'Timestamp':<26}  {'Model':<40}  Metadata")
    typer.echo("-" * 100)
    for c in ckpts:
        meta_str = ", ".join(f"{k}={v}" for k, v in c.metadata.items()) if c.metadata else ""
        typer.echo(
            f"{c.step:>8}  {c.timestamp:<26}  {c.model_name:<40}  {meta_str}"
        )


@checkpoint_app.command("inspect")
def checkpoint_inspect(
    path: str = typer.Argument(..., help="Path to a checkpoint directory."),
) -> None:
    """Show detailed checkpoint info."""
    import json
    from pathlib import Path

    meta_path = Path(path) / "meta.json"
    if not meta_path.exists():
        typer.echo(f"No meta.json found in {path}")
        raise typer.Exit(code=1)

    meta = json.loads(meta_path.read_text())
    typer.echo(json.dumps(meta, indent=2))


@checkpoint_app.command("export")
def checkpoint_export(
    path: str = typer.Argument(..., help="Checkpoint path."),
    output: str = typer.Option("./exported", help="Output directory."),
    format: str = typer.Option("lora", help="Export format: 'lora' or 'hf'."),
) -> None:
    """Export checkpoint weights."""
    import shutil
    from pathlib import Path

    src = Path(path)
    dst = Path(output)
    dst.mkdir(parents=True, exist_ok=True)

    if format == "lora":
        # Just copy adapter files
        for f in src.iterdir():
            if f.suffix in (".safetensors", ".bin", ".json") and f.name != "optimizer.pt":
                shutil.copy2(f, dst / f.name)
        typer.echo(f"LoRA adapter exported to {dst}")
    elif format == "hf":
        typer.echo(
            "Full HF export requires loading the model. Use:\n"
            "  from local_tinker.weights import merge_and_save\n"
            "  merge_and_save(training_client, output_path)"
        )
    else:
        typer.echo(f"Unknown format: {format}. Use 'lora' or 'hf'.")
        raise typer.Exit(code=1)
