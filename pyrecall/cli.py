"""pyrecall CLI — project management and snapshot inspection built with Typer."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from datetime import datetime, timedelta
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


def _parse_category_thresholds(raw: list[str]) -> dict[str, float]:
    """Parse ['safety=0.03', 'coding=0.15'] into {'safety': 0.03, 'coding': 0.15}."""
    result: dict[str, float] = {}
    for item in raw:
        if "=" not in item:
            raise typer.BadParameter(
                f"Expected 'category=value', got '{item}'",
                param_hint="--category-threshold",
            )
        cat, _, val = item.partition("=")
        try:
            result[cat.strip()] = float(val.strip())
        except ValueError:
            raise typer.BadParameter(
                f"Threshold value must be a number, got '{val}'",
                param_hint="--category-threshold",
            )
    return result


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
    category_threshold: Annotated[
        list[str],
        typer.Option(
            "--category-threshold",
            help="Per-category override in 'category=value' format, e.g. --category-threshold safety=0.03",
        ),
    ] = [],
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
    scoring_method: Annotated[
        str,
        typer.Option(
            "--scoring-method",
            help="Benchmark scoring method: 'log_likelihood' (recommended) or 'cosine' (legacy)",
        ),
    ] = "log_likelihood",
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
        errors.append(f"--threshold must be > 0 and <= 1, got {threshold}")
    try:
        parsed_category_thresholds = _parse_category_thresholds(category_threshold)
    except typer.BadParameter as exc:
        errors.append(str(exc))
        parsed_category_thresholds = {}
    for cat, val in parsed_category_thresholds.items():
        if not 0.0 < val <= 1.0:
            errors.append(f"--category-threshold {cat} must be between 0 and 1, got {val}")
    if replay_buffer_size < 0:
        errors.append(f"--replay-buffer-size must be >= 0, got {replay_buffer_size}")
    if not 0.0 <= replay_mix_ratio < 1.0:
        errors.append(f"--replay-mix-ratio must be in [0, 1), got {replay_mix_ratio}")
    if scoring_method not in ("log_likelihood", "cosine"):
        errors.append(
            f"--scoring-method must be 'log_likelihood' or 'cosine', got '{scoring_method}'"
        )
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
        "category_thresholds": parsed_category_thresholds,
        "replay_buffer_size": replay_buffer_size,
        "replay_mix_ratio": replay_mix_ratio,
        "scoring_method": scoring_method,
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
    gradient_checkpointing: Annotated[
        bool | None,
        typer.Option(
            "--gradient-checkpointing/--no-gradient-checkpointing",
            help="Enable gradient checkpointing to cut GPU memory ~40% at the cost of ~20% slower training.",
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
        scoring_method=config.get("scoring_method", "log_likelihood"),
        category_thresholds=config.get("category_thresholds", {}),
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
            gradient_checkpointing=gradient_checkpointing,
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
    compression: Annotated[
        str,
        typer.Option(
            "--compression",
            help="Compress adapter weights: 'none' (default), 'gzip', 'zstd', or 'lz4'.",
        ),
    ] = "none",
) -> None:
    """
    Load the model, run all benchmarks, and save a named capability snapshot.

    This is a slow operation — it runs benchmark prompts through the model
    and saves the LoRA adapter weights to disk.  Plan for several minutes on CPU.

    Pass --no-update-baseline to take the snapshot without overwriting the
    current baseline in .pyrecall.json.  Useful when you want to capture a
    point-in-time reading without disturbing your stable reference point.

    Use --compression gzip to reduce adapter storage by 40-60% (no extra deps).
    Use --compression zstd for faster compression with similar ratios (pip install zstandard).
    """
    from pyrecall.compress import SUPPORTED_CODECS

    if compression not in SUPPORTED_CODECS:
        console.print(
            f"[red]Error:[/red] Unknown compression '{compression}'. "
            f"Choose from: {sorted(SUPPORTED_CODECS)}"
        )
        raise typer.Exit(1)

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
        scoring_method=config.get("scoring_method", "log_likelihood"),
        snapshot_compression=compression,
        category_thresholds=config.get("category_thresholds", {}),
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
    ci: Annotated[
        bool,
        typer.Option("--ci", help="Machine-readable CI mode. Emits JSON and disables Rich output."),
    ] = False,
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
    category_threshold: Annotated[
        list[str],
        typer.Option(
            "--category-threshold",
            help="Per-category override in 'category=value' format, e.g. --category-threshold safety=0.03",
        ),
    ] = [],
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
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Save the report to a file. Format inferred from extension: .html, .md, or .json.",
        ),
    ] = None,
    watch: Annotated[
        bool,
        typer.Option(
            "--watch",
            "-w",
            help=(
                "Poll the snapshot directory and re-run the check whenever a new snapshot appears. "
                "Exits with code 2 if the last check detected forgetting, 0 otherwise. "
                "Stop with Ctrl-C."
            ),
        ),
    ] = False,
    interval: Annotated[
        int,
        typer.Option(
            "--interval",
            help="Seconds between polls when using --watch (default: 60).",
        ),
    ] = 60,
) -> None:
    """
    Compare two snapshots to detect forgotten skills.

    When called without arguments, compares the two most recently created
    snapshots.  Pass --before and --after to compare specific snapshots.
    Exits with code 2 when forgetting is detected.

    Use --json to get machine-readable output (includes per-prompt detail):

        pyrecall check --json | jq '.comparisons[].prompts'

    Use --output to save the report to a file:

        pyrecall check --output report.html
        pyrecall check --output report.md
        pyrecall check --output report.json

    Use --watch to poll continuously (useful during long training runs):

        pyrecall check --watch --interval 30
    """
    ts = ""
    n = 0
    config = _read_config()
    if ci:
        json_output = True
        verbose = False
    mgr = _build_rollback_manager(config)

    from pyrecall.detector import ForgettingDetector

    effective_threshold = (
        threshold if threshold is not None else config.get("forgetting_threshold", 0.10)
    )
    if not 0.0 < effective_threshold <= 1.0:
        if ci:
            typer.echo(f"Error: threshold must be between 0 and 1, got {effective_threshold}.")
        else:
            console.print(
                f"[red]Error:[/red] threshold must be between 0 and 1, got {effective_threshold}."
            )
        raise typer.Exit(1)
    effective_cat_thresholds = {
        **config.get("category_thresholds", {}),
        **_parse_category_thresholds(category_threshold),
    }
    detector = ForgettingDetector(
        threshold=effective_threshold, category_thresholds=effective_cat_thresholds
    )

    def _run_once() -> int:
        """Run a single check pass. Returns 0 (healthy) or 2 (forgetting detected)."""
        all_snaps = mgr.list_snapshots()
        if len(all_snaps) < 2:
            if ci:
                typer.echo(
                    "Error: Need at least two snapshots to run a forgetting check.\n"
                    "Run pyrecall snapshot <name> to create snapshots."
                )
            else:
                console.print(
                    "[red]Error:[/red] Need at least two snapshots to run a forgetting check.\n"
                    "Run [bold]pyrecall snapshot <name>[/bold] to create snapshots."
                )
            return 1

        if before is None and after is None:
            snap_before = all_snaps[-2]
            snap_after = all_snaps[-1]
        else:
            if before is None or after is None:
                if ci:
                    typer.echo("Error: Provide both --before and --after, or neither.")
                else:
                    console.print(
                        "[red]Error:[/red] Provide both --before and --after, or neither."
                    )
                return 1
            try:
                snap_before = mgr.load_snapshot(before)
            except FileNotFoundError:
                if ci:
                    typer.echo(f"Error: Snapshot '{before}' not found.")
                else:
                    console.print(f"[red]Error:[/red] Snapshot '{before}' not found.")
                return 1
            try:
                snap_after = mgr.load_snapshot(after)
            except FileNotFoundError:
                if ci:
                    typer.echo(f"Error: Snapshot '{after}' not found.")
                else:
                    console.print(f"[red]Error:[/red] Snapshot '{after}' not found.")
                return 1

        report = detector.compare(snap_before, snap_after)

        if json_output:
            typer.echo(report.to_json())
        else:
            report.print(verbose=verbose)

        if output:
            try:
                report.save(output)
                if ci:
                    typer.echo(f"Report saved to {output}")
                else:
                    console.print(f"[dim]Report saved to {output}[/dim]")
            except ValueError as exc:
                if ci:
                    typer.echo(f"Error: {exc}")
                else:
                    console.print(f"[red]Error:[/red] {exc}")
                return 1

        return 2 if report.degraded_skills else 0

    if not watch:
        raise typer.Exit(_run_once())

    # ── watch mode ─────────────────────────────────────────────────────────────
    if interval < 1:
        if ci:
            typer.echo("Error: --interval must be at least 1 second.")
        else:
            console.print("[red]Error:[/red] --interval must be at least 1 second.")
        raise typer.Exit(1)

    if ci:
        typer.echo(f"{ts} Waiting for a second snapshot ({n}/2)")
    else:
        console.print(f"[dim][{ts}][/dim] Waiting for a second snapshot ({n}/2)…")
    last_mtime: float | None = None
    last_exit_code = 0

    try:
        while True:
            # Compute a fingerprint for the snapshot directory state.
            try:
                current_mtime = max(
                    (p.stat().st_mtime for p in mgr.base_dir.rglob("snapshot.json")),
                    default=0.0,
                )
            except OSError as exc:
                if ci:
                    typer.echo(f"watch: could not stat snapshot directory: {exc}")
                else:
                    console.print(
                        f"[dim][yellow]watch: could not stat snapshot directory: {exc}[/yellow][/dim]"
                    )
                current_mtime = 0.0

            if current_mtime != last_mtime:
                last_mtime = current_mtime
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                all_snaps = mgr.list_snapshots()
                n = len(all_snaps)

                if n < 2:
                    if ci:
                        typer.echo(f"{ts} Waiting for a second snapshot ({n}/2)")
                    else:
                        console.print(f"[dim][{ts}][/dim] Waiting for a second snapshot ({n}/2)…")
                    last_exit_code = 0
                else:
                    # Default to the last two; override if named snapshots were given.
                    snap_b = all_snaps[-2]
                    snap_a = all_snaps[-1]
                    _failed = False

                    if before is not None:
                        try:
                            snap_b = mgr.load_snapshot(before)
                        except FileNotFoundError:
                            if ci:
                                typer.echo(f"Snapshot '{before}' not found.")
                            else:
                                console.print(
                                    f"[dim][{ts}][/dim] [red]Snapshot '{before}' not found.[/red]"
                                )
                            last_exit_code = 1
                            _failed = True

                    if after is not None:
                        try:
                            snap_a = mgr.load_snapshot(after)
                        except FileNotFoundError:
                            if ci:
                                typer.echo(f"Snapshot '{after}' not found.")
                            else:
                                console.print(
                                    f"[dim][{ts}][/dim] [red]Snapshot '{after}' not found.[/red]"
                                )
                            last_exit_code = 1
                            _failed = True

                    if _failed:
                        time.sleep(interval)
                        continue

                    report = detector.compare(snap_b, snap_a)
                    if report.degraded_skills:
                        cats = ", ".join(
                            f"{c} ({next((x.severity for x in report.comparisons if x.category == c), 'UNKNOWN')})"
                            for c in report.degraded_skills
                        )
                        if ci:
                            typer.echo(f"{ts} DEGRADED")
                        else:
                            console.print(
                                f"[dim][{ts}][/dim] [red]✗ DEGRADED[/red] — {snap_b.name} → {snap_a.name} | {cats}"
                            )
                        last_exit_code = 2
                    else:
                        if ci:
                            typer.echo(f"{ts} HEALTHY")
                        else:
                            console.print(
                                f"[dim][{ts}][/dim] [green]✓ healthy[/green] — {snap_b.name} → {snap_a.name} | {n} snapshots"
                            )
                        last_exit_code = 0

            time.sleep(interval)

    except KeyboardInterrupt:
        if ci:
            typer.echo("Watch stopped.")
        else:
            console.print("\n[dim]Watch stopped.[/dim]")

    raise typer.Exit(last_exit_code)


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
    category_threshold: Annotated[
        list[str],
        typer.Option(
            "--category-threshold",
            help="Per-category override in 'category=value' format, e.g. --category-threshold safety=0.03",
        ),
    ] = [],
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
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Save the report to a file. Format inferred from extension: .html, .md, or .json.",
        ),
    ] = None,
) -> None:
    """
    Diff two saved snapshots without running new benchmarks.

    Unlike 'check', this does not load the model or run any inference —
    it compares the stored benchmark scores directly.  Fast enough to run
    in any CI step.  Exits with code 2 when forgetting is detected.

        pyrecall diff before_v1 after_v2
        pyrecall diff before_v1 after_v2 --output report.html
        pyrecall diff before_v1 after_v2 --json | jq '.comparisons[].status'
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

    effective_cat_thresholds = {
        **config.get("category_thresholds", {}),
        **_parse_category_thresholds(category_threshold),
    }
    detector = ForgettingDetector(
        threshold=effective_threshold, category_thresholds=effective_cat_thresholds
    )
    report = detector.compare(snap_before, snap_after)

    if json_output:
        typer.echo(report.to_json())
    else:
        report.print(verbose=verbose)

    if output:
        try:
            report.save(output)
            console.print(f"[dim]Report saved to {output}[/dim]")
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)

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


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


@app.command()
def prune(
    snapshot_name: Annotated[
        str | None,
        typer.Argument(
            help="Name of a single snapshot to delete. Omit to use --keep-last or --older-than."
        ),
    ] = None,
    keep_last: Annotated[
        int | None,
        typer.Option(
            "--keep-last",
            help="Keep the N most recent snapshots; delete the rest.",
        ),
    ] = None,
    older_than: Annotated[
        int | None,
        typer.Option(
            "--older-than",
            help="Delete snapshots older than N days.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show what would be deleted without actually deleting anything.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Allow deleting the snapshot currently set as the baseline.",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt."),
    ] = False,
) -> None:
    """
    Delete old snapshots and free disk space.

    Three modes (pick one or combine --keep-last and --older-than):

        pyrecall prune before_v1              # delete one named snapshot
        pyrecall prune --keep-last 3          # keep 3 most recent, delete the rest
        pyrecall prune --older-than 30        # delete snapshots older than 30 days
        pyrecall prune --keep-last 5 --dry-run  # preview without deleting

    The current baseline is protected — pass --force to override.
    """
    config = _read_config()
    mgr = _build_rollback_manager(config)
    all_snaps = mgr.list_snapshots()
    baseline = config.get("baseline_snapshot")

    if not all_snaps:
        console.print("[yellow]No snapshots to prune.[/yellow]")
        return

    # ── resolve candidates ─────────────────────────────────────────────────────
    if snapshot_name is not None:
        # Single-name mode.
        if not mgr.has_snapshot(snapshot_name):
            console.print(
                f"[red]Error:[/red] Snapshot '{snapshot_name}' not found.\n"
                f"Available: {[s.name for s in all_snaps]}"
            )
            raise typer.Exit(1)
        candidates = [s for s in all_snaps if s.name == snapshot_name]
    else:
        candidates = list(all_snaps)

        if keep_last is not None:
            if keep_last < 0:
                console.print("[red]Error:[/red] --keep-last must be >= 0.")
                raise typer.Exit(1)
            # all_snaps is sorted oldest-first; keep the tail.
            keep_names = {s.name for s in all_snaps[-keep_last:]} if keep_last > 0 else set()
            candidates = [s for s in candidates if s.name not in keep_names]

        if older_than is not None:
            if older_than < 0:
                console.print("[red]Error:[/red] --older-than must be >= 0.")
                raise typer.Exit(1)
            cutoff = datetime.now() - timedelta(days=older_than)
            candidates = [s for s in candidates if s.created_at < cutoff]

        if keep_last is None and older_than is None:
            console.print(
                "[red]Error:[/red] Provide a snapshot name, --keep-last, or --older-than.\n"
                "Run [bold]pyrecall prune --help[/bold] for usage."
            )
            raise typer.Exit(1)

    if not candidates:
        console.print("[green]Nothing to prune.[/green]")
        return

    # ── baseline protection ────────────────────────────────────────────────────
    baseline_candidates = [s for s in candidates if s.name == baseline]
    if baseline_candidates and not force:
        console.print(
            f"[yellow]⚠  Snapshot '{baseline}' is the current baseline and is protected.[/yellow]\n"
            "Pass [bold]--force[/bold] to include it in the prune."
        )
        candidates = [s for s in candidates if s.name != baseline]
        if not candidates:
            console.print("[green]Nothing left to prune after baseline exclusion.[/green]")
            return

    # ── compute sizes ──────────────────────────────────────────────────────────
    snap_dirs = [mgr.base_dir / s.name for s in candidates]
    sizes = [_dir_size(d) if d.exists() else 0 for d in snap_dirs]
    total = sum(sizes)

    # ── preview ────────────────────────────────────────────────────────────────
    console.print()
    label = "[dim](dry run)[/dim] " if dry_run else ""
    console.print(f"{label}Snapshots to delete:\n")
    for snap, size in zip(candidates, sizes):
        baseline_note = " [dim](baseline)[/dim]" if snap.name == baseline else ""
        console.print(
            f"  [bold]{snap.name}[/bold]{baseline_note}  "
            f"[dim]{snap.created_at.strftime('%Y-%m-%d')}[/dim]  "
            f"{_human_size(size)}"
        )
    console.print(f"\n  Space to reclaim: [bold]{_human_size(total)}[/bold]")

    if dry_run:
        console.print("\n[dim]  Dry run — nothing deleted.[/dim]")
        return

    # ── confirm ────────────────────────────────────────────────────────────────
    if not yes:
        confirmed = typer.confirm("\nProceed?", default=False)
        if not confirmed:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(0)

    # ── delete ─────────────────────────────────────────────────────────────────
    deleted = 0
    for snap in candidates:
        try:
            mgr.delete_snapshot(snap.name)
            deleted += 1
            if snap.name == baseline:
                config["baseline_snapshot"] = None
                _write_config(config)
        except FileNotFoundError:
            console.print(f"[yellow]⚠  '{snap.name}' already gone — skipping.[/yellow]")

    console.print(
        f"\n[green]✓ Pruned {deleted} snapshot{'s' if deleted != 1 else ''}.[/green] "
        f"Freed ~{_human_size(total)}."
    )


@app.command()
def status(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output results as JSON instead of a rich table."),
    ] = False,
) -> None:
    """Show all saved snapshots and their per-category skill scores."""
    config = _read_config()
    mgr = _build_rollback_manager(config)
    all_snaps = mgr.list_snapshots()

    if not all_snaps:
        if json_output:
            typer.echo(json.dumps({"model_name": config.get("model_name"), "snapshots": []}))
        else:
            console.print(
                "[yellow]No snapshots found.[/yellow] "
                "Run [bold]pyrecall snapshot <name>[/bold] to create one."
            )
        return

    baseline = config.get("baseline_snapshot")

    if json_output:
        out = {
            "model_name": config.get("model_name"),
            "baseline_snapshot": baseline,
            "snapshots": [
                {
                    "name": snap.name,
                    "created_at": snap.created_at.isoformat(),
                    "overall": snap.overall_score(),
                    "scores": snap.category_scores(),
                    "adapter_ok": bool(snap.adapter_path and snap.adapter_path.exists()),
                    "is_baseline": snap.name == baseline,
                }
                for snap in all_snaps
            ],
        }
        typer.echo(json.dumps(out, indent=2))
        return

    # Collect all category names from any snapshot for column headers.
    all_categories: list[str] = []
    for snap in all_snaps:
        for cat in snap.category_scores():
            if cat not in all_categories:
                all_categories.append(cat)

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

        overall = snap.overall_score()
        overall_str = "-" if math.isnan(overall) else f"{overall:.3f}"
        row: list[str] = [
            name_markup,
            snap.created_at.strftime("%Y-%m-%d %H:%M"),
            overall_str,
        ]

        def _fmt_score(v: float) -> str:
            return "-" if math.isnan(v) else f"{v:.3f}"

        row += [_fmt_score(cat_scores[cat]) if cat in cat_scores else "-" for cat in all_categories]
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
    health: Annotated[
        bool,
        typer.Option(
            "--health",
            help=(
                "Show health status per snapshot instead of raw scores. "
                "Each snapshot is compared to the previous one using the configured forgetting threshold."
            ),
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output results as JSON instead of a rich table."),
    ] = False,
) -> None:
    """
    Show per-category score trends across all snapshots.

    Default view: each row is a snapshot; columns show scores per skill with a
    coloured trend arrow (↑ green = improved, ↓ red = dropped, → dim = unchanged).

    Use --health for a condensed view that shows whether each snapshot introduced
    forgetting versus the previous one, and which categories dropped.

        pyrecall history
        pyrecall history --health
        pyrecall history --last 5 --health
        pyrecall history --json
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
    baseline = config.get("baseline_snapshot")
    threshold = config.get("forgetting_threshold", 0.10)

    # ── health view ────────────────────────────────────────────────────────────
    if health or json_output:
        from pyrecall.detector import ForgettingDetector

        detector = ForgettingDetector(
            threshold=threshold,
            category_thresholds=config.get("category_thresholds", {}),
        )

        health_rows: list[dict] = []
        for i, snap in enumerate(snaps):
            is_baseline = snap.name == baseline
            if i == 0:
                health_rows.append(
                    {
                        "name": snap.name,
                        "created_at": snap.created_at.isoformat(),
                        "overall": round(snap.overall_score(), 4),
                        "status": "first",
                        "degraded_skills": [],
                        "notes": "(baseline)" if is_baseline else "(first snapshot)",
                        "is_baseline": is_baseline,
                    }
                )
            else:
                report = detector.compare(snaps[i - 1], snap)
                _comp_map = {x.category: x for x in report.comparisons}
                dropped_notes = [
                    f"{c} {_comp_map[c].delta:+.3f}"
                    for c in report.degraded_skills
                    if c in _comp_map
                ]
                health_rows.append(
                    {
                        "name": snap.name,
                        "created_at": snap.created_at.isoformat(),
                        "overall": round(snap.overall_score(), 4),
                        "status": "degraded" if report.degraded_skills else "healthy",
                        "degraded_skills": report.degraded_skills,
                        "notes": ", ".join(dropped_notes) if dropped_notes else "",
                        "is_baseline": is_baseline,
                    }
                )

        if json_output:
            typer.echo(
                json.dumps(
                    {
                        "model": config["model_name"],
                        "threshold": threshold,
                        "snapshots": health_rows,
                    },
                    indent=2,
                )
            )
            return

        # Render health table.
        table = Table(title=f"Snapshot Health Timeline — {config['model_name']}", show_lines=False)
        table.add_column("Snapshot", style="bold white", no_wrap=True)
        table.add_column("Created", style="dim", no_wrap=True)
        table.add_column("Overall", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Notes", no_wrap=False)

        for hr in health_rows:
            name_str = (
                f"[bold green]{hr['name']} ★[/bold green]" if hr["is_baseline"] else hr["name"]
            )
            if hr["status"] == "first":
                status_str = "[dim]—[/dim]"
            elif hr["status"] == "healthy":
                status_str = "[green]✓ healthy[/green]"
            else:
                status_str = "[red]✗ DEGRADED[/red]"

            notes = hr["notes"]
            if hr["is_baseline"] and hr["status"] != "first":
                notes = (notes + " ← baseline").strip() if notes else "← baseline"

            table.add_row(
                name_str,
                hr["created_at"][:16].replace("T", " "),
                f"{hr['overall']:.3f}",
                status_str,
                notes,
            )

        console.print(table)
        if baseline:
            console.print(f"[dim]  ★ = current baseline ({baseline})[/dim]")
        return

    # ── score trend view (default) ─────────────────────────────────────────────

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
            if cat not in cat_scores:
                row.append("-")
                continue
            score = cat_scores[cat]
            cell = f"{score:.3f}"
            if prev is not None and cat in prev:
                cell += f" {_trend(prev[cat], score)}"
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

    entries = next((s["count"] for s in mgr.suites() if s["name"] == registered), 0)
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


@benchmark_app.command("validate")
def benchmark_validate(
    name: Annotated[str, typer.Argument(help="Name of the custom suite to validate")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output results as JSON instead of formatted report."),
    ] = False,
) -> None:
    """
    Run static quality checks on a custom benchmark suite without loading a model.

    Checks prompt length, reference answer length, duplicate prompts, category
    balance, and reference answer variety. No inference is performed.

        pyrecall benchmark validate my_suite
        pyrecall benchmark validate my_suite --json

    Exit code 0 if no errors (warnings are non-blocking); 1 if errors found.
    """
    from collections import Counter

    from pyrecall.benchmarks.custom import CustomBenchmarkManager, _parse_jsonl

    mgr = CustomBenchmarkManager()
    suite_path = mgr.base_dir / f"{name}.jsonl"

    if not suite_path.exists():
        from pyrecall.benchmarks.default import CATEGORIES

        if name in CATEGORIES:
            console.print(
                f"[dim]'{name}' is a built-in benchmark category — no validation needed.[/dim]"
            )
            return
        available = [s["name"] for s in mgr.suites()]
        console.print(
            f"[red]Error:[/red] Suite '{name}' not found.\n"
            f"Available: {available or ['(none registered)']}"
        )
        raise typer.Exit(1)

    entries = _parse_jsonl(suite_path)
    if not entries:
        console.print(f"[red]Error:[/red] Suite '{name}' is empty or contains no valid entries.")
        raise typer.Exit(1)

    prompts = [e["prompt"] for e in entries]
    refs = [e["reference_answer"] for e in entries]
    cats = [e.get("category", name) for e in entries]

    errors: list[str] = []
    warnings: list[str] = []
    infos: list[str] = []

    # ERROR: prompt too short (< 10 tokens by whitespace split)
    for e in entries:
        p = e["prompt"]
        if len(p.split()) < 10:
            errors.append(f'Prompt too short ({len(p.split())} tokens): "{p[:60]}"')

    # ERROR: reference answer too short (< 3 tokens)
    for e in entries:
        r = e["reference_answer"]
        if len(r.split()) < 3:
            errors.append(
                f'Reference answer too short ({len(r.split())} tokens): "{r}" '
                f'(prompt: "{e["prompt"][:40]}")'
            )

    # ERROR: duplicate prompts within the suite
    seen: set[str] = set()
    for p in prompts:
        if p in seen:
            errors.append(f'Duplicate prompt: "{p[:60]}"')
        seen.add(p)

    # WARNING: any category with only 1 prompt (Cohen's d unavailable)
    cat_counts = Counter(cats)
    for cat, count in cat_counts.items():
        if count == 1:
            warnings.append(
                f"Category '{cat}' has only 1 prompt — Cohen's d will not be computed for it."
            )

    # WARNING: all reference answers identical
    if len(set(refs)) == 1:
        warnings.append(
            "All reference answers are identical — suite may not differentiate model quality."
        )

    # INFO: prompts not ending with a sentence-ending character
    odd = [p for p in prompts if not p.rstrip().endswith(("?", ":", "."))]
    if odd:
        infos.append(
            f"{len(odd)} prompt(s) don't end with '?', ':', or '.' — "
            "verify they are complete sentences."
        )

    n_cats = len(cat_counts)

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "suite": name,
                    "prompts": len(entries),
                    "categories": n_cats,
                    "errors": errors,
                    "warnings": warnings,
                    "info": infos,
                    "valid": len(errors) == 0,
                },
                indent=2,
            )
        )
    else:
        cat_label = "categories" if n_cats != 1 else "category"
        console.print(
            f"\nValidating suite '[bold]{name}[/bold]' "
            f"({len(entries)} prompt{'s' if len(entries) != 1 else ''} "
            f"across {n_cats} {cat_label})…\n"
        )
        for msg in errors:
            console.print(f"  [red]✗[/red]  {msg}")
        for msg in warnings:
            console.print(f"  [yellow]⚠[/yellow]  {msg}")
        for msg in infos:
            console.print(f"  [dim]ℹ[/dim]  {msg}")
        if not errors and not warnings and not infos:
            console.print("  [green]✓[/green]  All checks passed.")

        parts = []
        if errors:
            parts.append(f"[red]{len(errors)} error{'s' if len(errors) != 1 else ''}[/red]")
        if warnings:
            parts.append(
                f"[yellow]{len(warnings)} warning{'s' if len(warnings) != 1 else ''}[/yellow]"
            )
        if infos:
            parts.append(f"[dim]{len(infos)} info[/dim]")
        console.print(f"\nResult: {', '.join(parts) if parts else '[green]clean[/green]'}.")
        if errors:
            console.print("[dim]  Fix errors before using this suite in model.snapshot().[/dim]\n")

    if errors:
        raise typer.Exit(1)


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

        def _safe_round(v: float, n: int) -> float | None:
            return None if math.isnan(v) else round(v, n)

        records = [
            {
                "name": snap.name,
                "created_at": snap.created_at.isoformat(),
                "overall": _safe_round(snap.overall_score(), 4),
                "categories": {
                    cat: _safe_round(score, 4) for cat, score in snap.category_scores().items()
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
            overall = snap.overall_score()
            row: dict = {
                "snapshot": snap.name,
                "created_at": snap.created_at.isoformat(),
                "overall": "" if math.isnan(overall) else round(overall, 4),
            }
            for cat in all_categories:
                v = cat_scores.get(cat)
                row[cat] = "" if v is None or math.isnan(v) else round(v, 4)
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
