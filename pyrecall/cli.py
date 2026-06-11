"""pyrecall CLI — project management and snapshot inspection built with Typer."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="pyrecall",
    help=(
        "pyrecall — continuous fine-tuning with automatic forgetting detection.\n\n"
        "Quickstart:\n\n"
        "  pyrecall init --model meta-llama/Llama-3.2-1B\n\n"
        "  # take a snapshot before training\n"
        "  pyrecall snapshot before_v1\n\n"
        "  # ... run your training script ...\n\n"
        "  pyrecall status   # inspect all snapshots\n"
        "  pyrecall check    # compare last two snapshots\n"
        "  pyrecall rollback before_v1  # if forgetting is detected"
    ),
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()

_CONFIG_FILE = ".pyrecall.json"


# ── helpers ────────────────────────────────────────────────────────────────────


def _read_config() -> dict:
    cfg_path = Path(_CONFIG_FILE)
    if not cfg_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] No {_CONFIG_FILE} found in the current directory.\n"
            "Run [bold]pyrecall init[/bold] first."
        )
        raise typer.Exit(1)
    try:
        return json.loads(cfg_path.read_text())
    except json.JSONDecodeError as exc:
        console.print(
            f"[bold red]Error:[/bold red] {_CONFIG_FILE} is not valid JSON: {exc}\n"
            "Fix or delete it and run [bold]pyrecall init[/bold] again."
        )
        raise typer.Exit(1) from exc


def _write_config(data: dict) -> None:
    Path(_CONFIG_FILE).write_text(json.dumps(data, indent=2))


def _build_rollback_manager(config: dict):
    from pyrecall.rollback import RollbackManager

    return RollbackManager(model_name=config["model_name"])


# ── commands ───────────────────────────────────────────────────────────────────


@app.command()
def init(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="HuggingFace model identifier"),
    ] = "meta-llama/Llama-3.2-1B",
    strategy: Annotated[
        str,
        typer.Option("--strategy", "-s", help="Fine-tuning strategy: 'lora' or 'qlora'"),
    ] = "lora",
    lora_r: Annotated[
        int,
        typer.Option("--lora-r", help="LoRA rank"),
    ] = 16,
    lora_alpha: Annotated[
        int,
        typer.Option("--lora-alpha", help="LoRA scaling factor (typically 2× rank)"),
    ] = 32,
    lora_dropout: Annotated[
        float,
        typer.Option("--lora-dropout", help="LoRA dropout rate"),
    ] = 0.1,
    learning_rate: Annotated[
        float,
        typer.Option("--learning-rate", help="AdamW learning rate for fine-tuning"),
    ] = 2e-4,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Per-device training batch size"),
    ] = 4,
    max_length: Annotated[
        int,
        typer.Option("--max-length", help="Tokenisation truncation length"),
    ] = 512,
    threshold: Annotated[
        float,
        typer.Option("--threshold", help="Score drop fraction that counts as forgetting (0–1)"),
    ] = 0.10,
    replay_buffer_size: Annotated[
        int,
        typer.Option("--replay-buffer-size", help="Max past examples stored for replay (0 = disabled)"),
    ] = 500,
    replay_mix_ratio: Annotated[
        float,
        typer.Option("--replay-mix-ratio", help="Fraction of each training batch filled with replayed examples (0–1)"),
    ] = 0.3,
) -> None:
    """Initialise pyrecall in the current project directory."""
    errors: list[str] = []
    if strategy not in ("lora", "qlora"):
        errors.append(f"--strategy must be 'lora' or 'qlora', got '{strategy}'")
    if lora_r < 1:
        errors.append(f"--lora-r must be >= 1, got {lora_r}")
    if not 0.0 <= lora_dropout < 1.0:
        errors.append(f"--lora-dropout must be in [0, 1), got {lora_dropout}")
    if learning_rate <= 0:
        errors.append(f"--learning-rate must be > 0, got {learning_rate}")
    if batch_size < 1:
        errors.append(f"--batch-size must be >= 1, got {batch_size}")
    if max_length < 1:
        errors.append(f"--max-length must be >= 1, got {max_length}")
    if not 0.0 < threshold <= 1.0:
        errors.append(f"--threshold must be between 0 and 1, got {threshold}")
    if replay_buffer_size < 0:
        errors.append(f"--replay-buffer-size must be >= 0, got {replay_buffer_size}")
    if not 0.0 <= replay_mix_ratio < 1.0:
        errors.append(f"--replay-mix-ratio must be in [0, 1), got {replay_mix_ratio}")
    if errors:
        for msg in errors:
            console.print(f"[red]Error:[/red] {msg}")
        raise typer.Exit(1)

    cfg_path = Path(_CONFIG_FILE)
    if cfg_path.exists():
        console.print(
            f"[yellow]⚠  {_CONFIG_FILE} already exists.[/yellow] "
            "Delete it first to reinitialise."
        )
        raise typer.Exit(1)

    config = {
        "model_name": model,
        "strategy": strategy,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_length": max_length,
        "forgetting_threshold": threshold,
        "replay_buffer_size": replay_buffer_size,
        "replay_mix_ratio": replay_mix_ratio,
        "created_at": datetime.now().isoformat(),
        "baseline_snapshot": None,
    }
    _write_config(config)

    console.print(f"[green]✓ Initialised pyrecall[/green] with [bold]{model}[/bold] ({strategy})")
    console.print(f"[dim]  Config saved to {_CONFIG_FILE}[/dim]")
    console.print()
    console.print("Next steps:")
    console.print("  [bold]pyrecall snapshot before_v1[/bold]   — take a baseline snapshot")
    console.print("  [bold]pyrecall status[/bold]               — view all snapshots")


@app.command()
def snapshot(
    name: Annotated[str, typer.Argument(help="Name for this snapshot, e.g. 'before_v2'")],
) -> None:
    """
    Load the model, run all benchmarks, and save a named capability snapshot.

    This is a slow operation — it runs 20 benchmark prompts through the model
    and saves the LoRA adapter weights to disk.  Plan for several minutes on CPU.
    """
    config = _read_config()

    from pyrecall.model import Model

    model_obj = Model(
        config["model_name"],
        strategy=config.get("strategy", "lora"),
        lora_r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.1),
        learning_rate=config.get("learning_rate", 2e-4),
        batch_size=config.get("batch_size", 4),
        max_length=config.get("max_length", 512),
        forgetting_threshold=config.get("forgetting_threshold", 0.10),
        replay_buffer_size=config.get("replay_buffer_size", 500),
        replay_mix_ratio=config.get("replay_mix_ratio", 0.3),
    )
    model_obj.snapshot(name=name)

    config["baseline_snapshot"] = name
    _write_config(config)

    console.print(
        f"[dim]  Baseline updated to '{name}' in {_CONFIG_FILE}.[/dim]"
    )


@app.command()
def check(
    before: Annotated[
        Optional[str],
        typer.Option("--before", help="Snapshot name to use as baseline"),
    ] = None,
    after: Annotated[
        Optional[str],
        typer.Option("--after", help="Snapshot name to compare against"),
    ] = None,
    threshold: Annotated[
        Optional[float],
        typer.Option("--threshold", help="Override the forgetting threshold (0–1). Defaults to the value set in pyrecall init."),
    ] = None,
) -> None:
    """
    Compare two snapshots to detect forgotten skills.

    When called without arguments, compares the two most recently created
    snapshots.  Pass --before and --after to compare specific snapshots.
    Exits with code 2 when forgetting is detected.
    """
    config = _read_config()
    mgr = _build_rollback_manager(config)
    all_snaps = mgr.list_snapshots()

    if len(all_snaps) < 2:
        console.print(
            "[red]Error:[/red] Need at least two snapshots to run a forgetting check.\n"
            "Run [bold]pyrecall snapshot <name>[/bold] to create snapshots."
        )
        raise typer.Exit(1)

    if before is None and after is None:
        # Compare the last two chronologically.
        snap_before = all_snaps[-2]
        snap_after = all_snaps[-1]
    else:
        if before is None or after is None:
            console.print(
                "[red]Error:[/red] Provide both --before and --after, or neither."
            )
            raise typer.Exit(1)
        try:
            snap_before = mgr.load_snapshot(before)
        except FileNotFoundError:
            console.print(f"[red]Error:[/red] Snapshot '{before}' not found.")
            raise typer.Exit(1)
        try:
            snap_after = mgr.load_snapshot(after)
        except FileNotFoundError:
            console.print(f"[red]Error:[/red] Snapshot '{after}' not found.")
            raise typer.Exit(1)

    from pyrecall.detector import ForgettingDetector

    effective_threshold = threshold if threshold is not None else config.get("forgetting_threshold", 0.10)
    if not 0.0 < effective_threshold <= 1.0:
        console.print(
            f"[red]Error:[/red] threshold must be between 0 and 1, got {effective_threshold}."
        )
        raise typer.Exit(1)
    detector = ForgettingDetector(threshold=effective_threshold)
    report = detector.compare(snap_before, snap_after)
    report.print()

    if report.degraded_skills:
        raise typer.Exit(2)  # Non-zero exit so CI pipelines can catch forgetting.


@app.command()
def rollback(
    snapshot_name: Annotated[
        str, typer.Argument(help="Snapshot to roll back to")
    ],
) -> None:
    """
    Update the project config to point at a previous snapshot.

    This does not reload the model in memory — it updates .pyrecall.json so that
    the next Python session loading Model() will start from this snapshot's
    adapter weights via model.rollback(to='<name>').

    To rollback immediately in a running session, call model.rollback() in Python.
    """
    config = _read_config()
    mgr = _build_rollback_manager(config)

    if not mgr.has_snapshot(snapshot_name):
        available = [s.name for s in mgr.list_snapshots()]
        console.print(
            f"[red]Error:[/red] Snapshot '{snapshot_name}' not found.\n"
            f"Available: {available}"
        )
        raise typer.Exit(1)

    old_baseline = config.get("baseline_snapshot")
    config["baseline_snapshot"] = snapshot_name
    _write_config(config)

    console.print(
        f"[green]✓ Baseline updated[/green]: "
        f"'{old_baseline}' → '[bold]{snapshot_name}[/bold]'"
    )
    console.print(
        f"[dim]  To apply in Python: model.rollback(to='{snapshot_name}')[/dim]"
    )


@app.command()
def delete(
    snapshot_name: Annotated[
        str, typer.Argument(help="Snapshot to permanently delete")
    ],
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """
    Permanently delete a snapshot and its adapter weights.

    This cannot be undone.  Pass --yes to skip the confirmation prompt,
    which is useful in non-interactive scripts and CI pipelines.
    """
    config = _read_config()
    mgr = _build_rollback_manager(config)

    if not mgr.has_snapshot(snapshot_name):
        available = [s.name for s in mgr.list_snapshots()]
        console.print(
            f"[bold red]Error:[/bold red] Snapshot '{snapshot_name}' not found.\n"
            f"Available: {available}"
        )
        raise typer.Exit(1)

    if not yes:
        confirmed = typer.confirm(
            f"Permanently delete snapshot '{snapshot_name}' and its adapter weights?",
            default=False,
        )
        if not confirmed:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(0)

    mgr.delete_snapshot(snapshot_name)

    was_baseline = config.get("baseline_snapshot") == snapshot_name
    if was_baseline:
        config["baseline_snapshot"] = None
        _write_config(config)
        console.print(
            f"[green]✓ Deleted '{snapshot_name}'.[/green] "
            "[dim]It was the current baseline — baseline cleared.[/dim]"
        )
    else:
        console.print(f"[green]✓ Deleted snapshot '{snapshot_name}'.[/green]")


@app.command()
def status() -> None:
    """Show all saved snapshots and their per-category skill scores."""
    config = _read_config()
    mgr = _build_rollback_manager(config)
    all_snaps = mgr.list_snapshots()

    if not all_snaps:
        console.print(
            "[yellow]No snapshots found.[/yellow] "
            "Run [bold]pyrecall snapshot <name>[/bold] to create one."
        )
        return

    # Collect all category names from any snapshot for column headers.
    all_categories: list[str] = []
    for snap in all_snaps:
        for cat in snap.category_scores():
            if cat not in all_categories:
                all_categories.append(cat)

    baseline = config.get("baseline_snapshot")
    table = Table(
        title=f"Snapshots — {config['model_name']}",
        show_lines=False,
    )
    table.add_column("Name", style="bold white")
    table.add_column("Created", style="dim")
    table.add_column("Overall", justify="right")
    for cat in all_categories:
        table.add_column(cat.replace("_", " ").title(), justify="right")
    table.add_column("Adapter", justify="center")

    for snap in all_snaps:
        cat_scores = snap.category_scores()
        is_baseline = snap.name == baseline
        name_markup = f"[bold green]{snap.name} ★[/bold green]" if is_baseline else snap.name
        adapter_ok = "✓" if (snap.adapter_path and snap.adapter_path.exists()) else "✗"

        row: list[str] = [
            name_markup,
            snap.created_at.strftime("%Y-%m-%d %H:%M"),
            f"{snap.overall_score():.3f}",
        ]
        row += [f"{cat_scores.get(cat, 0.0):.3f}" for cat in all_categories]
        row.append(adapter_ok)
        table.add_row(*row)

    console.print(table)
    if baseline:
        console.print(f"[dim]  ★ = current baseline ({baseline})[/dim]")
