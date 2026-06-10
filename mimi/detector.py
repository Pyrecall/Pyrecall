"""ForgettingDetector — compare two snapshots and surface degraded skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO

from rich.console import Console
from rich.table import Table

from .snapshot import SkillSnapshot
from .utils import console as _shared_console


@dataclass
class CategoryComparison:
    """Before/after scores for one skill category."""

    category: str
    score_before: float
    score_after: float

    @property
    def delta(self) -> float:
        """Absolute change in score (positive = improved, negative = degraded)."""
        return self.score_after - self.score_before

    @property
    def pct_change(self) -> float:
        """Percentage change relative to the before score."""
        if self.score_before == 0.0:
            return 0.0
        return (self.score_after - self.score_before) / self.score_before * 100.0


@dataclass
class ForgettingReport:
    """
    Result of a forgetting check.

    Contains per-category comparisons and exposes helpers for printing
    and programmatic inspection.
    """

    snapshot_before: str
    snapshot_after: str
    threshold: float
    comparisons: list[CategoryComparison] = field(default_factory=list)

    # ── inspection ─────────────────────────────────────────────────────────────

    @property
    def degraded_skills(self) -> list[str]:
        """Categories whose score dropped more than *threshold*."""
        return [
            c.category
            for c in self.comparisons
            if (c.score_before - c.score_after) > self.threshold
        ]

    @property
    def is_healthy(self) -> bool:
        """True when no skill degraded beyond the threshold."""
        return len(self.degraded_skills) == 0

    # ── rendering ──────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        buf = StringIO()
        self._render(Console(file=buf, highlight=False))
        return buf.getvalue()

    def print(self) -> None:
        """Print the report to the terminal using rich formatting."""
        self._render(_shared_console)

    def _render(self, console: Console) -> None:
        table = Table(
            title=(
                f"Forgetting Report  [dim]{self.snapshot_before}[/dim]"
                f" → [dim]{self.snapshot_after}[/dim]"
            ),
            show_lines=False,
        )
        table.add_column("Skill", style="bold white")
        table.add_column("Before", justify="right")
        table.add_column("After", justify="right")
        table.add_column("Δ Score", justify="right")
        table.add_column("Status", justify="center")

        for comp in self.comparisons:
            degraded = (comp.score_before - comp.score_after) > self.threshold
            sign = "+" if comp.delta >= 0 else ""
            delta_str = f"{sign}{comp.delta:.3f} ({sign}{comp.pct_change:.1f}%)"
            delta_style = "red" if comp.delta < -self.threshold else (
                "green" if comp.delta >= 0 else "yellow"
            )
            status_markup = "[red]FORGOTTEN[/red]" if degraded else "[green]  OK  [/green]"

            table.add_row(
                comp.category,
                f"{comp.score_before:.3f}",
                f"{comp.score_after:.3f}",
                f"[{delta_style}]{delta_str}[/{delta_style}]",
                status_markup,
            )

        console.print()
        console.print(table)

        if self.degraded_skills:
            console.print(
                f"\n[error]⚠  Forgetting detected in: {', '.join(self.degraded_skills)}[/error]"
            )
            console.print(
                "[dim]  Run model.rollback(to='<snapshot>') to restore these skills.[/dim]\n"
            )
        else:
            console.print(
                "\n[success]✓  No significant forgetting detected "
                f"(threshold: {self.threshold:.0%}).[/success]\n"
            )


class ForgettingDetector:
    """
    Compare a before-snapshot and an after-snapshot to detect forgotten skills.

    A skill is considered *forgotten* when its average cosine-similarity score
    drops by more than *threshold* (default 10 percentage points).
    """

    def __init__(self, threshold: float = 0.10) -> None:
        self.threshold = threshold

    def compare(
        self, before: SkillSnapshot, after: SkillSnapshot
    ) -> ForgettingReport:
        """
        Return a ForgettingReport comparing *before* and *after* snapshots.

        Categories present in only one snapshot get a score of 0.0 for the missing side.
        """
        before_scores = before.category_scores()
        after_scores = after.category_scores()

        all_categories = sorted(set(before_scores) | set(after_scores))
        comparisons = [
            CategoryComparison(
                category=cat,
                score_before=before_scores.get(cat, 0.0),
                score_after=after_scores.get(cat, 0.0),
            )
            for cat in all_categories
        ]

        return ForgettingReport(
            snapshot_before=before.name,
            snapshot_after=after.name,
            threshold=self.threshold,
            comparisons=comparisons,
        )
