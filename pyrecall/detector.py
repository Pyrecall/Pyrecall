"""ForgettingDetector — compare two snapshots and surface degraded skills."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import StringIO

from rich.console import Console
from rich.table import Table

from .snapshot import SkillSnapshot
from .utils import console as _shared_console


@dataclass
class PromptComparison:
    """Before/after scores for a single benchmark prompt."""

    category: str
    prompt: str
    score_before: float
    score_after: float

    @property
    def delta(self) -> float:
        return self.score_after - self.score_before

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "prompt": self.prompt,
            "score_before": round(self.score_before, 4),
            "score_after": round(self.score_after, 4),
            "delta": round(self.delta, 4),
        }


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
    category_thresholds: dict[str, float] = field(default_factory=dict)
    comparisons: list[CategoryComparison] = field(default_factory=list)
    prompt_comparisons: list[PromptComparison] = field(default_factory=list)

    def _threshold_for(self, category: str) -> float:
        """Return the effective threshold for *category*, falling back to the global default."""
        return self.category_thresholds.get(category, self.threshold)

    # ── inspection ─────────────────────────────────────────────────────────────

    @property
    def degraded_skills(self) -> list[str]:
        """Categories whose score dropped more than their effective threshold."""
        return [
            c.category
            for c in self.comparisons
            if (c.score_before - c.score_after) > self._threshold_for(c.category)
        ]

    @property
    def is_healthy(self) -> bool:
        """True when no skill degraded beyond the threshold."""
        return len(self.degraded_skills) == 0

    def prompts_for_category(self, category: str) -> list[PromptComparison]:
        """Return per-prompt comparisons for *category*, worst delta first."""
        return sorted(
            [p for p in self.prompt_comparisons if p.category == category],
            key=lambda p: p.delta,
        )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of the report."""
        return {
            "healthy": self.is_healthy,
            "snapshot_before": self.snapshot_before,
            "snapshot_after": self.snapshot_after,
            "threshold": self.threshold,
            "degraded_skills": self.degraded_skills,
            "comparisons": [
                {
                    "category": c.category,
                    "score_before": round(c.score_before, 4),
                    "score_after": round(c.score_after, 4),
                    "delta": round(c.delta, 4),
                    "pct_change": round(c.pct_change, 2),
                    "threshold": self._threshold_for(c.category),
                    "status": "FORGOTTEN"
                    if (c.score_before - c.score_after) > self._threshold_for(c.category)
                    else "OK",
                    "prompts": [p.to_dict() for p in self.prompts_for_category(c.category)],
                }
                for c in self.comparisons
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Return the report serialised as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    # ── rendering ──────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        buf = StringIO()
        self._render(Console(file=buf, highlight=False))
        return buf.getvalue()

    def print(self, verbose: bool = False) -> None:
        """Print the report to the terminal using rich formatting."""
        self._render(_shared_console, verbose=verbose)

    def _render(self, console: Console, verbose: bool = False) -> None:
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
            cat_threshold = self._threshold_for(comp.category)
            degraded = (comp.score_before - comp.score_after) > cat_threshold
            sign = "+" if comp.delta >= 0 else ""
            delta_str = f"{sign}{comp.delta:.3f} ({sign}{comp.pct_change:.1f}%)"
            delta_style = (
                "red" if comp.delta < -cat_threshold else ("green" if comp.delta >= 0 else "yellow")
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
            threshold_note = (
                f"(threshold: {self.threshold:.0%}"
                + (", with per-category overrides" if self.category_thresholds else "")
                + ")"
            )
            console.print(
                f"\n[success]✓  No significant forgetting detected {threshold_note}.[/success]\n"
            )

        if verbose and self.prompt_comparisons:
            categories_to_show = (
                self.degraded_skills
                if self.degraded_skills
                else sorted({p.category for p in self.prompt_comparisons})
            )
            for cat in categories_to_show:
                prompts = self.prompts_for_category(cat)
                if not prompts:
                    continue
                pt = Table(
                    title=f"[bold]{cat}[/bold] — per-prompt breakdown",
                    show_lines=False,
                    title_justify="left",
                )
                pt.add_column("Prompt", no_wrap=False, max_width=60)
                pt.add_column("Before", justify="right")
                pt.add_column("After", justify="right")
                pt.add_column("Δ", justify="right")
                for p in prompts:
                    sign = "+" if p.delta >= 0 else ""
                    delta_style = "red" if p.delta < 0 else "green"
                    pt.add_row(
                        p.prompt,
                        f"{p.score_before:.3f}",
                        f"{p.score_after:.3f}",
                        f"[{delta_style}]{sign}{p.delta:.3f}[/{delta_style}]",
                    )
                console.print(pt)
                console.print()


class ForgettingDetector:
    """
    Compare a before-snapshot and an after-snapshot to detect forgotten skills.

    A skill is considered *forgotten* when its average cosine-similarity score
    drops by more than its effective threshold.  The global *threshold* applies
    to all categories unless overridden in *category_thresholds*.

    Example::

        detector = ForgettingDetector(
            threshold=0.10,
            category_thresholds={"safety": 0.03, "coding": 0.15},
        )
    """

    def __init__(
        self,
        threshold: float = 0.10,
        category_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.threshold = threshold
        self.category_thresholds: dict[str, float] = category_thresholds or {}

    def compare(self, before: SkillSnapshot, after: SkillSnapshot) -> ForgettingReport:
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

        # Build per-prompt comparisons by matching on (category, prompt) key.
        before_map = {(s.category, s.prompt): s.score for s in before.scores}
        after_map = {(s.category, s.prompt): s.score for s in after.scores}
        all_keys = sorted(set(before_map) | set(after_map))
        prompt_comparisons = [
            PromptComparison(
                category=cat,
                prompt=prompt,
                score_before=before_map.get((cat, prompt), 0.0),
                score_after=after_map.get((cat, prompt), 0.0),
            )
            for cat, prompt in all_keys
        ]

        return ForgettingReport(
            snapshot_before=before.name,
            snapshot_after=after.name,
            threshold=self.threshold,
            category_thresholds=self.category_thresholds,
            comparisons=comparisons,
            prompt_comparisons=prompt_comparisons,
        )
