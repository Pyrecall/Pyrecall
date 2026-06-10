"""mimi CLI — project management and snapshot inspection built with Typer."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="mimi",
    help=(
        "mimi — continuous fine-tuning with automatic forgetting detection.\n\n"
        "Quickstart:\n\n"
        "  mimi init --model meta-llama/Llama-3.2-1B\n\n"
        "  # take a snapshot before training\n"
        "  mimi snapshot before_v1\n\n"
        "  # ... run your training script ...\n\n"
        "  mimi status   # inspect all snapshots\n"
        "  mimi check    # compare last two snapshots\n"
        "  mimi rollback before_v1  # if forgetting is detected"
    ),
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()

_CONFIG_FILE = ".mimi.json"


# ── helpers ────────────────────────────────────────────────────────────────────


def _read_config() -> dict:
    cfg_path = Path(_CONFIG_FILE)
    if not cfg_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] No {_CONFIG_FILE} found in the current directory.\n"
            "Run [bold]mimi init[/bold] first."
        )
        raise typer.Exit(1)
    return json.loads(cfg_path.read_text())


def _write_config(data: dict) -> None:
    Path(_CONFIG_FILE).write_text(json.dumps(data, indent=2))


def _build_rollback_manager(config: dict):
    from mimi.rollback import RollbackManager

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
        typer.Option("--strategy", "-s", help="Fine-tuning strategy (only 'lora' supported)"),
    ] = "lora",
) -> None:
    """Initialise mimi in the current project directory."""
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
        "created_at": datetime.now().isoformat(),
        "baseline_snapshot": None,
    }
    _write_config(config)

    console.print(f"[green]✓ Initialised mimi[/green] with [bold]{model}[/bold] ({strategy})")
    console.print(f"[dim]  Config saved to {_CONFIG_FILE}[/dim]")
    console.print()
    console.print("Next steps:")
    console.print("  [bold]mimi snapshot before_v1[/bold]   — take a baseline snapshot")
    console.print("  [bold]mimi status[/bold]               — view all snapshots")


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

    from mimi.model import Model

    model_obj = Model(config["model_name"], strategy=config.get("strategy", "lora"))
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
) -> None:
    """
    Compare two snapshots to detect forgotten skills.

    When called without arguments, compares the two most recently created
    snapshots.  Pass --before and --after to compare specific snapshots.
    """
    config = _read_config()
    mgr = _build_rollback_manager(config)
    all_snaps = mgr.list_snapshots()

    if len(all_snaps) < 2:
        console.print(
            "[red]Error:[/red] Need at least two snapshots to run a forgetting check.\n"
            "Run [bold]mimi snapshot <name>[/bold] to create snapshots."
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
        snap_before = mgr.load_snapshot(before)
        snap_after = mgr.load_snapshot(after)

    from mimi.detector import ForgettingDetector

    detector = ForgettingDetector(threshold=0.10)
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

    This does not reload the model in memory — it updates .mimi.json so that
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
def status() -> None:
    """Show all saved snapshots and their per-category skill scores."""
    config = _read_config()
    mgr = _build_rollback_manager(config)
    all_snaps = mgr.list_snapshots()

    if not all_snaps:
        console.print(
            "[yellow]No snapshots found.[/yellow] "
            "Run [bold]mimi snapshot <name>[/bold] to create one."
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
