"""pyrecall CLI — project management and snapshot inspection built with Typer."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

try:
    from importlib.metadata import version as _pkg_version

    _VERSION = _pkg_version("pyrecall")
except Exception:
    _VERSION = "unknown"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"pyrecall {_VERSION}")
        raise typer.Exit()


app = typer.Typer(
    name="pyrecall",
    help=(
        "pyrecall — continuous fine-tuning with automatic forgetting detection.\n\n"
        "Quickstart:\n\n"
        "  pyrecall init --model meta-llama/Llama-3.2-1B\n\n"
        "  # take a snapshot before training\n"
        "  pyrecall snapshot before_v1\n\n"
        "  # fine-tune on new data\n"
        "  pyrecall learn train.jsonl --epochs 3 --snapshot-after after_v1\n\n"
        "  pyrecall status   # inspect all snapshots\n"
        "  pyrecall check    # compare last two snapshots\n"
        "  pyrecall rollback before_v1  # if forgetting is detected"
    ),
    add_completion=False,
    rich_markup_mode="rich",
)

replay_app = typer.Typer(
    name="replay",
    help="Inspect and manage the replay buffer.",
    add_completion=False,
    rich_markup_mode="rich",
)
app.add_typer(replay_app, name="replay")

live_app = typer.Typer(
    name="live",
    help="Inspect and manage the live-learning interaction database.",
    add_completion=False,
    rich_markup_mode="rich",
)
app.add_typer(live_app, name="live")

benchmark_app = typer.Typer(
    name="benchmark",
    help="Manage custom benchmark suites.",
    add_completion=False,
    rich_markup_mode="rich",
)
app.add_typer(benchmark_app, name="benchmark")


@app.callback()
def _main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ] = None,
) -> None:
    pass


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


def _build_trackers(
    log_wandb: bool, log_mlflow: bool, log_neptune: bool = False, neptune_project: str | None = None
):
    trackers: list = []
    if log_wandb:
        from pyrecall.trackers import WandbTracker

        trackers.append(WandbTracker())
    if log_mlflow:
        from pyrecall.trackers import MLflowTracker

        trackers.append(MLflowTracker())
    if log_neptune:
        from pyrecall.trackers import NeptuneTracker

        if not neptune_project:
            raise typer.BadParameter(
                "--neptune-project is required when --log-neptune is set",
                param_hint="--neptune-project",
            )
        trackers.append(NeptuneTracker(project=neptune_project))
    return trackers if trackers else None


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
        typer.Option(
            "--replay-buffer-size", help="Max past examples stored for replay (0 = disabled)"
        ),
    ] = 500,
    replay_mix_ratio: Annotated[
        float,
        typer.Option(
            "--replay-mix-ratio",
            help="Fraction of each training batch filled with replayed examples (0–1)",
        ),
    ] = 0.3,
) -> None:
    """Initialise pyrecall in the current project directory."""
    errors: list[str] = []
    if not model or " " in model or model.startswith("/"):
        errors.append(
            f"--model must be a HuggingFace model ID (e.g. 'meta-llama/Llama-3.2-1B'), got '{model}'"
        )
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
            f"[yellow]⚠  {_CONFIG_FILE} already exists.[/yellow] Delete it first to reinitialise."
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
def learn(
    data: Annotated[
        str,
        typer.Argument(
            help="Path to training data (.jsonl, .csv, or .parquet). Each row needs a 'text' column."
        ),
    ],
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Number of full passes over the training data"),
    ] = 3,
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size", help="Per-device training batch size (overrides init setting)"
        ),
    ] = None,
    learning_rate: Annotated[
        float | None,
        typer.Option("--learning-rate", help="AdamW learning rate (overrides init setting)"),
    ] = None,
    max_length: Annotated[
        int | None,
        typer.Option(
            "--max-length", help="Tokenisation truncation length (overrides init setting)"
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume", help="Resume from the latest checkpoint if a previous run was interrupted"
        ),
    ] = False,
    snapshot_before: Annotated[
        str | None,
        typer.Option("--snapshot-before", help="Take a named snapshot before training begins"),
    ] = None,
    snapshot_after: Annotated[
        str | None,
        typer.Option(
            "--snapshot-after", help="Take a named snapshot immediately after training completes"
        ),
    ] = None,
    no_update_baseline: Annotated[
        bool,
        typer.Option(
            "--no-update-baseline",
            help="Do not update baseline_snapshot in .pyrecall.json after snapshotting",
        ),
    ] = False,
    log_wandb: Annotated[
        bool,
        typer.Option(
            "--log-wandb",
            help="Log snapshot scores to Weights & Biases (requires pip install pyrecall[wandb])",
        ),
    ] = False,
    log_mlflow: Annotated[
        bool,
        typer.Option(
            "--log-mlflow",
            help="Log snapshot scores to MLflow (requires pip install pyrecall[mlflow])",
        ),
    ] = False,
    log_neptune: Annotated[
        bool,
        typer.Option(
            "--log-neptune",
            help="Log snapshot scores to Neptune (requires pip install pyrecall[neptune])",
        ),
    ] = False,
    neptune_project: Annotated[
        str | None,
        typer.Option(
            "--neptune-project",
            help="Neptune project in 'workspace/project' format (required with --log-neptune)",
        ),
    ] = None,
) -> None:
    """
    Fine-tune the model on a local dataset.

    Reads hyperparameters from .pyrecall.json unless overridden by flags.
    Use --snapshot-before and --snapshot-after to bracket training with
    snapshots so you can immediately run pyrecall check:

        pyrecall learn train.jsonl \\
            --snapshot-before before_v1 \\
            --snapshot-after after_v1
        pyrecall check

    Pass --no-update-baseline to keep your existing baseline unchanged even
    when --snapshot-before or --snapshot-after are used.  This is useful in
    CI where you want a stable reference point regardless of training outcome.
    """
    if not Path(data).exists():
        console.print(f"[red]Error:[/red] Training data file not found: '{data}'")
        raise typer.Exit(1)

    config = _read_config()

    from pyrecall.model import Model, PyrecallError

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

    tracker = _build_trackers(log_wandb, log_mlflow, log_neptune, neptune_project)

    if snapshot_before:
        model_obj.snapshot(name=snapshot_before, tracker=tracker)
        if not no_update_baseline:
            config["baseline_snapshot"] = snapshot_before
            _write_config(config)
            console.print(f"[dim]  Baseline set to '{snapshot_before}' in {_CONFIG_FILE}.[/dim]")
        else:
            console.print(f"[dim]  Snapshot '{snapshot_before}' taken (baseline unchanged).[/dim]")

    try:
        model_obj.learn(
            data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            resume=resume,
        )
    except PyrecallError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if snapshot_after:
        model_obj.snapshot(name=snapshot_after, tracker=tracker)
        if not no_update_baseline:
            config["baseline_snapshot"] = snapshot_after
            _write_config(config)
            console.print(f"[dim]  Baseline updated to '{snapshot_after}' in {_CONFIG_FILE}.[/dim]")
        else:
            console.print(f"[dim]  Snapshot '{snapshot_after}' taken (baseline unchanged).[/dim]")


@app.command()
def snapshot(
    name: Annotated[str, typer.Argument(help="Name for this snapshot, e.g. 'before_v2'")],
    no_update_baseline: Annotated[
        bool,
        typer.Option(
            "--no-update-baseline", help="Do not update baseline_snapshot in .pyrecall.json"
        ),
    ] = False,
    log_wandb: Annotated[
        bool,
        typer.Option(
            "--log-wandb",
            help="Log scores to Weights & Biases (requires pip install pyrecall[wandb])",
        ),
    ] = False,
    log_mlflow: Annotated[
        bool,
        typer.Option(
            "--log-mlflow", help="Log scores to MLflow (requires pip install pyrecall[mlflow])"
        ),
    ] = False,
    log_neptune: Annotated[
        bool,
        typer.Option(
            "--log-neptune",
            help="Log scores to Neptune (requires pip install pyrecall[neptune])",
        ),
    ] = False,
    neptune_project: Annotated[
        str | None,
        typer.Option(
            "--neptune-project",
            help="Neptune project in 'workspace/project' format (required with --log-neptune)",
        ),
    ] = None,
) -> None:
    """
    Load the model, run all benchmarks, and save a named capability snapshot.

    This is a slow operation — it runs 64 benchmark prompts through the model
    and saves the LoRA adapter weights to disk.  Plan for several minutes on CPU.

    Pass --no-update-baseline to take the snapshot without overwriting the
    current baseline in .pyrecall.json.  Useful when you want to capture a
    point-in-time reading without disturbing your stable reference point.
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
    tracker = _build_trackers(log_wandb, log_mlflow, log_neptune, neptune_project)
    model_obj.snapshot(name=name, tracker=tracker)

    if not no_update_baseline:
        config["baseline_snapshot"] = name
        _write_config(config)
        console.print(f"[dim]  Baseline updated to '{name}' in {_CONFIG_FILE}.[/dim]")
    else:
        console.print(f"[dim]  Snapshot '{name}' taken (baseline unchanged).[/dim]")


@app.command()
def check(
    before: Annotated[
        str | None,
        typer.Option("--before", help="Snapshot name to use as baseline"),
    ] = None,
    after: Annotated[
        str | None,
        typer.Option("--after", help="Snapshot name to compare against"),
    ] = None,
    threshold: Annotated[
        float | None,
        typer.Option(
            "--threshold",
            help="Override the forgetting threshold (0–1). Defaults to the value set in pyrecall init.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output results as JSON instead of a rich table. Useful for CI pipelines and dashboards.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v", help="Show per-prompt score breakdown for each degraded skill."
        ),
    ] = False,
) -> None:
    """
    Compare two snapshots to detect forgotten skills.

    When called without arguments, compares the two most recently created
    snapshots.  Pass --before and --after to compare specific snapshots.
    Exits with code 2 when forgetting is detected.

    Use --json to get machine-readable output (includes per-prompt detail):

        pyrecall check --json | jq '.comparisons[].prompts'

    Use --verbose to see which specific benchmark prompts drove a drop:

        pyrecall check --verbose
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
            console.print("[red]Error:[/red] Provide both --before and --after, or neither.")
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

    effective_threshold = (
        threshold if threshold is not None else config.get("forgetting_threshold", 0.10)
    )
    if not 0.0 < effective_threshold <= 1.0:
        console.print(
            f"[red]Error:[/red] threshold must be between 0 and 1, got {effective_threshold}."
        )
        raise typer.Exit(1)
    detector = ForgettingDetector(threshold=effective_threshold)
    report = detector.compare(snap_before, snap_after)

    if json_output:
        typer.echo(report.to_json())
    else:
        report.print(verbose=verbose)

    if report.degraded_skills:
        raise typer.Exit(2)  # Non-zero exit so CI pipelines can catch forgetting.


@app.command()
def diff(
    snap1: Annotated[str, typer.Argument(help="Name of the 'before' snapshot")],
    snap2: Annotated[str, typer.Argument(help="Name of the 'after' snapshot")],
    threshold: Annotated[
        float | None,
        typer.Option(
            "--threshold",
            help="Override the forgetting threshold (0–1). Defaults to the value set in pyrecall init.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output results as JSON instead of a rich table.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v", help="Show per-prompt score breakdown for each degraded skill."
        ),
    ] = False,
) -> None:
    """
    Diff two saved snapshots without running new benchmarks.

    Unlike 'check', this does not load the model or run any inference —
    it compares the stored benchmark scores directly.  Fast enough to run
    in any CI step.  Exits with code 2 when forgetting is detected.

        pyrecall diff before_v1 after_v2
        pyrecall diff before_v1 after_v2 --json | jq '.comparisons[].status'
        pyrecall diff before_v1 after_v2 --verbose
    """
    config = _read_config()
    mgr = _build_rollback_manager(config)

    try:
        snap_before = mgr.load_snapshot(snap1)
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Snapshot '{snap1}' not found.")
        raise typer.Exit(1)
    try:
        snap_after = mgr.load_snapshot(snap2)
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Snapshot '{snap2}' not found.")
        raise typer.Exit(1)

    from pyrecall.detector import ForgettingDetector

    effective_threshold = (
        threshold if threshold is not None else config.get("forgetting_threshold", 0.10)
    )
    if not 0.0 < effective_threshold <= 1.0:
        console.print(
            f"[red]Error:[/red] threshold must be between 0 and 1, got {effective_threshold}."
        )
        raise typer.Exit(1)

    detector = ForgettingDetector(threshold=effective_threshold)
    report = detector.compare(snap_before, snap_after)

    if json_output:
        typer.echo(report.to_json())
    else:
        report.print(verbose=verbose)

    if report.degraded_skills:
        raise typer.Exit(2)


@app.command()
def compare(
    snapshots: Annotated[
        list[str],
        typer.Argument(help="Two or more snapshot names to compare side by side"),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output results as JSON instead of a rich table."),
    ] = False,
) -> None:
    """
    Compare N snapshots side by side in a single table.

    Unlike 'diff' (which is limited to two snapshots), 'compare' accepts any
    number of names and renders every category score as columns so you can
    see the full progression across multiple training runs at a glance.

    The best score in each row is highlighted green; the worst is red.

        pyrecall compare before_v1 after_v1 after_v2 after_v3
        pyrecall compare before_v1 after_v1 --json
    """
    if len(snapshots) < 2:
        console.print("[red]Error:[/red] At least two snapshot names are required.")
        raise typer.Exit(1)

    config = _read_config()
    mgr = _build_rollback_manager(config)

    loaded: list = []
    for name in snapshots:
        try:
            loaded.append(mgr.load_snapshot(name))
        except FileNotFoundError:
            console.print(f"[red]Error:[/red] Snapshot '{name}' not found.")
            raise typer.Exit(1)

    # Collect all category names in a stable order.
    all_cats: list[str] = []
    for snap in loaded:
        for cat in snap.category_scores():
            if cat not in all_cats:
                all_cats.append(cat)

    if json_output:
        out: dict = {
            "snapshots": [s.name for s in loaded],
            "categories": {
                "overall": {s.name: round(s.overall_score(), 4) for s in loaded},
                **{
                    cat: {s.name: round(s.category_scores().get(cat, 0.0), 4) for s in loaded}
                    for cat in all_cats
                },
            },
        }
        typer.echo(json.dumps(out, indent=2))
        return

    table = Table(
        title=f"Snapshot Comparison — {config['model_name']}",
        show_lines=False,
    )
    table.add_column("Category", style="bold white")
    for snap in loaded:
        table.add_column(snap.name, justify="right")

    def _fmt_row(label: str, values: list[float]) -> None:
        best = max(values)
        worst = min(values)
        cells: list[str] = [label]
        for v in values:
            s = f"{v:.3f}"
            if v == best and best != worst:
                cells.append(f"[green]{s}[/green]")
            elif v == worst and best != worst:
                cells.append(f"[red]{s}[/red]")
            else:
                cells.append(s)
        table.add_row(*cells)

    _fmt_row("overall", [s.overall_score() for s in loaded])
    for cat in all_cats:
        _fmt_row(cat, [s.category_scores().get(cat, 0.0) for s in loaded])

    console.print(table)


@app.command()
def rollback(
    snapshot_name: Annotated[str, typer.Argument(help="Snapshot to roll back to")],
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
            f"[red]Error:[/red] Snapshot '{snapshot_name}' not found.\nAvailable: {available}"
        )
        raise typer.Exit(1)

    old_baseline = config.get("baseline_snapshot")
    config["baseline_snapshot"] = snapshot_name
    _write_config(config)

    console.print(
        f"[green]✓ Baseline updated[/green]: '{old_baseline}' → '[bold]{snapshot_name}[/bold]'"
    )
    console.print(f"[dim]  To apply in Python: model.rollback(to='{snapshot_name}')[/dim]")


@app.command()
def delete(
    snapshot_name: Annotated[str, typer.Argument(help="Snapshot to permanently delete")],
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


@app.command()
def history(
    category: Annotated[
        str | None,
        typer.Option("--category", "-c", help="Show trend for a single category only"),
    ] = None,
    last: Annotated[
        int,
        typer.Option("--last", "-n", help="Limit to the N most recent snapshots"),
    ] = 0,
) -> None:
    """
    Show per-category score trends across all snapshots.

    Each row is a snapshot; columns show scores per skill with a coloured
    trend arrow (↑ green = improved, ↓ red = dropped, → dim = unchanged).
    Use --last N to focus on the most recent snapshots.
    """
    config = _read_config()
    mgr = _build_rollback_manager(config)
    all_snaps = mgr.list_snapshots()

    if not all_snaps:
        console.print(
            "[yellow]No snapshots found.[/yellow] "
            "Run [bold]pyrecall snapshot <name>[/bold] to create one."
        )
        return

    if len(all_snaps) < 2:
        console.print(
            "[yellow]Only one snapshot found — need at least two to show trends.[/yellow]\n"
            "Take another snapshot after training to see how scores change."
        )
        return

    snaps = all_snaps[-last:] if last > 0 else all_snaps

    # Determine which categories to show.
    all_categories: list[str] = []
    for snap in snaps:
        for cat in snap.category_scores():
            if cat not in all_categories:
                all_categories.append(cat)

    if category:
        if category not in all_categories:
            console.print(
                f"[red]Error:[/red] Category '{category}' not found. Available: {all_categories}"
            )
            raise typer.Exit(1)
        display_categories = [category]
    else:
        display_categories = all_categories

    table = Table(
        title=f"Score History — {config['model_name']}",
        show_lines=True,
    )
    table.add_column("Snapshot", style="bold white", no_wrap=True)
    table.add_column("Date", style="dim", no_wrap=True)
    table.add_column("Overall", justify="right")
    for cat in display_categories:
        table.add_column(cat.replace("_", " ").title(), justify="right")

    def _trend(prev: float, curr: float) -> str:
        delta = curr - prev
        if delta > 0.005:
            return "[green]↑[/green]"
        if delta < -0.005:
            return "[red]↓[/red]"
        return "[dim]→[/dim]"

    baseline = config.get("baseline_snapshot")

    for i, snap in enumerate(snaps):
        cat_scores = snap.category_scores()
        prev = snaps[i - 1].category_scores() if i > 0 else None
        prev_overall = snaps[i - 1].overall_score() if i > 0 else None

        overall_str = f"{snap.overall_score():.3f}"
        if prev_overall is not None:
            overall_str += f" {_trend(prev_overall, snap.overall_score())}"

        name_markup = (
            f"[bold green]{snap.name} ★[/bold green]" if snap.name == baseline else snap.name
        )

        row: list[str] = [
            name_markup,
            snap.created_at.strftime("%Y-%m-%d %H:%M"),
            overall_str,
        ]
        for cat in display_categories:
            score = cat_scores.get(cat, 0.0)
            cell = f"{score:.3f}"
            if prev is not None:
                cell += f" {_trend(prev.get(cat, score), score)}"
            row.append(cell)

        table.add_row(*row)

    console.print(table)

    # Summary line: overall drift from first to last shown snapshot.
    first_overall = snaps[0].overall_score()
    last_overall = snaps[-1].overall_score()
    delta = last_overall - first_overall
    direction = (
        "[green]improved[/green]"
        if delta > 0
        else "[red]dropped[/red]"
        if delta < 0
        else "unchanged"
    )
    console.print(
        f"\n  Overall score {direction} by [bold]{abs(delta):.3f}[/bold] "
        f"across {len(snaps)} snapshots "
        f"({snaps[0].name} → {snaps[-1].name})."
    )
    if baseline:
        console.print(f"[dim]  ★ = current baseline ({baseline})[/dim]")


# ── benchmark subcommands ──────────────────────────────────────────────────────


@benchmark_app.command("add")
def benchmark_add(
    path: Annotated[str, typer.Argument(help="Path to a .jsonl benchmark file")],
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Suite name (defaults to the filename stem)"),
    ] = None,
) -> None:
    """
    Register a custom benchmark suite.

    The file must be JSONL with one benchmark per line.  Each line needs at
    least a 'prompt' and a 'reference_answer' key.  An optional 'category'
    key labels the skill; if omitted the suite name is used instead.

        pyrecall benchmark add nautical.jsonl
        pyrecall benchmark add domain.jsonl --name my_domain

    Example line:

        {"prompt": "What does port mean on a ship?", "reference_answer": "The left side when facing the bow.", "category": "nautical"}
    """
    from pyrecall.benchmarks.custom import CustomBenchmarkManager

    mgr = CustomBenchmarkManager()
    try:
        registered = mgr.add(path, name=name)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    entries = next(s["count"] for s in mgr.suites() if s["name"] == registered)
    console.print(
        f"[green]✓[/green] Registered benchmark suite [bold]{registered}[/bold] "
        f"({entries} prompt{'s' if entries != 1 else ''})."
    )


@benchmark_app.command("list")
def benchmark_list() -> None:
    """Show all registered custom benchmark suites."""
    from pyrecall.benchmarks.custom import CustomBenchmarkManager

    mgr = CustomBenchmarkManager()
    suites = mgr.suites()

    if not suites:
        console.print(
            "[yellow]No custom benchmarks registered.[/yellow] "
            "Run [bold]pyrecall benchmark add <file.jsonl>[/bold] to add one."
        )
        return

    table = Table(title="Custom Benchmark Suites", show_lines=False)
    table.add_column("Name", style="bold white")
    table.add_column("Prompts", justify="right")
    table.add_column("Path", style="dim")

    for suite in suites:
        table.add_row(suite["name"], str(suite["count"]), suite["path"])

    console.print(table)
    total = sum(s["count"] for s in suites)
    console.print(
        f"[dim]  {total} custom prompt{'s' if total != 1 else ''} across {len(suites)} suite{'s' if len(suites) != 1 else ''}.[/dim]"
    )


@benchmark_app.command("remove")
def benchmark_remove(
    name: Annotated[str, typer.Argument(help="Name of the suite to remove")],
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """Remove a registered custom benchmark suite."""
    from pyrecall.benchmarks.custom import CustomBenchmarkManager

    mgr = CustomBenchmarkManager()

    if not yes:
        confirmed = typer.confirm(f"Remove benchmark suite '{name}'?")
        if not confirmed:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(0)

    try:
        mgr.remove(name)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] Removed benchmark suite [bold]{name}[/bold].")


# ── replay subcommands ─────────────────────────────────────────────────────────


@replay_app.command("status")
def replay_status() -> None:
    """Show the current state of the replay buffer for this project's model."""
    config = _read_config()

    from pyrecall.replay import ReplayBuffer

    model_name = config["model_name"]
    max_size = config.get("replay_buffer_size", 500)

    if max_size == 0:
        console.print("[yellow]Replay buffer is disabled[/yellow] (replay_buffer_size = 0).")
        console.print(
            "Re-run [bold]pyrecall init[/bold] with [bold]--replay-buffer-size > 0[/bold] to enable it."
        )
        return

    buf = ReplayBuffer(model_name, max_size=max_size)
    filled = len(buf)
    pct = filled / max_size * 100 if max_size else 0
    bar_width = 30
    filled_bars = int(bar_width * filled / max_size) if max_size else 0
    bar = "█" * filled_bars + "░" * (bar_width - filled_bars)

    console.print(f"\n  Model    [bold]{model_name}[/bold]")
    console.print(f"  Buffer   [{bar}] {filled}/{max_size} ({pct:.0f}%)")
    console.print(f"  Seen     {buf.total_seen} total examples added since creation\n")

    if filled == 0:
        console.print("[dim]  Buffer is empty — run pyrecall learn to populate it.[/dim]")


@replay_app.command("clear")
def replay_clear(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """Permanently wipe the replay buffer for this project's model."""
    config = _read_config()

    from pyrecall.replay import ReplayBuffer

    model_name = config["model_name"]
    max_size = config.get("replay_buffer_size", 500)
    buf = ReplayBuffer(model_name, max_size=max_size)

    if len(buf) == 0:
        console.print("[dim]Replay buffer is already empty.[/dim]")
        return

    if not yes:
        confirmed = typer.confirm(
            f"Permanently clear {len(buf)} examples from the replay buffer for '{model_name}'?",
            default=False,
        )
        if not confirmed:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(0)

    buf.clear()
    console.print(f"[green]✓ Replay buffer cleared[/green] for [bold]{model_name}[/bold].")


# ── export ─────────────────────────────────────────────────────────────────────


@app.command()
def export(
    output: Annotated[
        str | None,
        typer.Argument(help="Output file path (.csv or .json). Omit to print JSON to stdout."),
    ] = None,
    fmt: Annotated[
        str | None,
        typer.Option(
            "--format",
            "-f",
            help="Force output format: 'csv' or 'json'. Auto-detected from file extension when omitted.",
        ),
    ] = None,
) -> None:
    """
    Export all snapshot scores to CSV or JSON for external analysis.

    CSV produces one row per snapshot per category (tidy/long format),
    ready to load into pandas or a spreadsheet:

        pyrecall export scores.csv

    JSON produces one object per snapshot with nested category scores:

        pyrecall export scores.json

    Omit the output path to stream JSON to stdout:

        pyrecall export | jq '.[0].categories'
    """
    config = _read_config()
    mgr = _build_rollback_manager(config)
    all_snaps = mgr.list_snapshots()

    if not all_snaps:
        console.print(
            "[yellow]No snapshots found.[/yellow] "
            "Run [bold]pyrecall snapshot <name>[/bold] to create one."
        )
        return

    # Resolve format.
    resolved_fmt = fmt
    if resolved_fmt is None:
        if output:
            ext = Path(output).suffix.lower()
            if ext == ".csv":
                resolved_fmt = "csv"
            elif ext in (".json", ".jsonl"):
                resolved_fmt = "json"
            else:
                console.print(
                    f"[red]Error:[/red] Cannot infer format from extension '{ext}'. "
                    "Use --format csv or --format json."
                )
                raise typer.Exit(1)
        else:
            resolved_fmt = "json"

    if resolved_fmt not in ("csv", "json"):
        console.print(f"[red]Error:[/red] Unknown format '{resolved_fmt}'. Use 'csv' or 'json'.")
        raise typer.Exit(1)

    if resolved_fmt == "json":
        records = [
            {
                "name": snap.name,
                "created_at": snap.created_at.isoformat(),
                "overall": round(snap.overall_score(), 4),
                "categories": {
                    cat: round(score, 4) for cat, score in snap.category_scores().items()
                },
            }
            for snap in all_snaps
        ]
        payload = json.dumps(records, indent=2)
        if output:
            Path(output).write_text(payload)
            console.print(
                f"[green]✓ Exported {len(all_snaps)} snapshots to[/green] [bold]{output}[/bold]"
            )
        else:
            sys.stdout.write(payload + "\n")

    else:  # csv
        all_categories: list[str] = []
        for snap in all_snaps:
            for cat in snap.category_scores():
                if cat not in all_categories:
                    all_categories.append(cat)

        fieldnames = ["snapshot", "created_at", "overall"] + all_categories
        rows = []
        for snap in all_snaps:
            cat_scores = snap.category_scores()
            row: dict = {
                "snapshot": snap.name,
                "created_at": snap.created_at.isoformat(),
                "overall": round(snap.overall_score(), 4),
            }
            for cat in all_categories:
                row[cat] = round(cat_scores.get(cat, 0.0), 4)
            rows.append(row)

        if output:
            with open(output, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            console.print(
                f"[green]✓ Exported {len(all_snaps)} snapshots to[/green] [bold]{output}[/bold]"
            )
        else:
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


# ── live subcommands ───────────────────────────────────────────────────────────


def _default_live_db() -> Path:
    return Path.home() / ".pyrecall" / "live_data.db"


@live_app.command("status")
def live_status() -> None:
    """Show statistics for the live-learning interaction database."""
    db_path = _default_live_db()

    if not db_path.exists():
        console.print(
            "[yellow]No live-learning database found.[/yellow]\n"
            "Start recording interactions via [bold]LiveLearner.record()[/bold] "
            "or [bold]model.serve()[/bold] to populate it."
        )
        return

    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN trained = 0 THEN 1 ELSE 0 END) AS pending, "
            "SUM(CASE WHEN trained = 1 THEN 1 ELSE 0 END) AS trained "
            "FROM interactions"
        ).fetchone()
        total: int = row["total"] or 0
        pending: int = row["pending"] or 0
        trained: int = row["trained"] or 0

        newest = conn.execute(
            "SELECT timestamp FROM interactions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        oldest = conn.execute(
            "SELECT timestamp FROM interactions ORDER BY id ASC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    console.print(f"\n  Database  [bold]{db_path}[/bold]")
    console.print(f"  Total     {total} interactions")
    console.print(f"  Pending   {pending} (not yet used for training)")
    console.print(f"  Trained   {trained}")
    if oldest:
        console.print(f"  Oldest    {oldest['timestamp']}")
    if newest:
        console.print(f"  Newest    {newest['timestamp']}")
    console.print()

    if pending == 0 and total == 0:
        console.print("[dim]  Database is empty.[/dim]")


@live_app.command("clear")
def live_clear(
    all_: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Delete ALL interactions including already-trained ones (default: pending only)",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """
    Clear interactions from the live-learning database.

    By default only pending (untrained) interactions are removed.
    Pass --all to wipe the entire database including trained rows.
    """
    db_path = _default_live_db()

    if not db_path.exists():
        console.print("[dim]No live-learning database found — nothing to clear.[/dim]")
        return

    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if all_:
            count = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        else:
            count = conn.execute("SELECT COUNT(*) FROM interactions WHERE trained = 0").fetchone()[
                0
            ]
    finally:
        conn.close()

    if count == 0:
        label = "interactions" if all_ else "pending interactions"
        console.print(f"[dim]No {label} to clear.[/dim]")
        return

    scope = "ALL interactions (including trained)" if all_ else f"{count} pending interactions"
    if not yes:
        confirmed = typer.confirm(f"Permanently delete {scope}?", default=False)
        if not confirmed:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(0)

    conn = sqlite3.connect(db_path)
    try:
        if all_:
            conn.execute("DELETE FROM interactions")
        else:
            conn.execute("DELETE FROM interactions WHERE trained = 0")
        conn.commit()
    finally:
        conn.close()

    label = "all interactions" if all_ else f"{count} pending interactions"
    console.print(f"[green]✓ Cleared {label}[/green] from live-learning database.")
