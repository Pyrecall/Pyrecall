"""Tests for ForgettingDetector and ForgettingReport."""

from __future__ import annotations

from datetime import datetime

import pytest

from mimi.detector import CategoryComparison, ForgettingDetector, ForgettingReport
from mimi.snapshot import SkillScore, SkillSnapshot


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_snapshot(name: str, scores: dict[str, float]) -> SkillSnapshot:
    """Build a SkillSnapshot with one SkillScore per category using the given score."""
    skill_scores = [
        SkillScore(
            category=cat,
            prompt=f"Test prompt for {cat}",
            response=f"Test response for {cat}",
            score=val,
        )
        for cat, val in scores.items()
    ]
    return SkillSnapshot(
        name=name,
        model_name="test/model",
        created_at=datetime(2024, 1, 1),
        scores=skill_scores,
    )


# ── CategoryComparison ────────────────────────────────────────────────────────


class TestCategoryComparison:
    def test_delta_positive_when_improved(self) -> None:
        c = CategoryComparison(category="reasoning", score_before=0.5, score_after=0.7)
        assert c.delta == pytest.approx(0.2, abs=1e-6)

    def test_delta_negative_when_degraded(self) -> None:
        c = CategoryComparison(category="coding", score_before=0.8, score_after=0.5)
        assert c.delta == pytest.approx(-0.3, abs=1e-6)

    def test_pct_change_zero_when_before_is_zero(self) -> None:
        c = CategoryComparison(category="safety", score_before=0.0, score_after=0.5)
        assert c.pct_change == 0.0

    def test_pct_change_correct(self) -> None:
        c = CategoryComparison(category="reasoning", score_before=0.8, score_after=0.6)
        assert c.pct_change == pytest.approx(-25.0, abs=0.1)


# ── ForgettingReport ──────────────────────────────────────────────────────────


class TestForgettingReport:
    def _make_report(
        self, comparisons: list[tuple[str, float, float]], threshold: float = 0.10
    ) -> ForgettingReport:
        return ForgettingReport(
            snapshot_before="before",
            snapshot_after="after",
            threshold=threshold,
            comparisons=[
                CategoryComparison(category=cat, score_before=b, score_after=a)
                for cat, b, a in comparisons
            ],
        )

    def test_is_healthy_when_no_degradation(self) -> None:
        report = self._make_report(
            [("reasoning", 0.8, 0.85), ("coding", 0.7, 0.72)]
        )
        assert report.is_healthy is True

    def test_degraded_skills_empty_when_healthy(self) -> None:
        report = self._make_report([("reasoning", 0.8, 0.80)])
        assert report.degraded_skills == []

    def test_detects_forgotten_skill(self) -> None:
        report = self._make_report(
            [("coding", 0.80, 0.60)],  # drop of 0.20 > threshold 0.10
            threshold=0.10,
        )
        assert "coding" in report.degraded_skills
        assert report.is_healthy is False

    def test_does_not_flag_small_drop(self) -> None:
        report = self._make_report(
            [("reasoning", 0.80, 0.72)],  # drop of 0.08 < threshold 0.10
            threshold=0.10,
        )
        assert report.degraded_skills == []

    def test_degraded_skills_contains_only_bad_categories(self) -> None:
        report = self._make_report(
            [
                ("reasoning", 0.80, 0.85),   # improved
                ("coding", 0.80, 0.60),       # forgotten
                ("safety", 0.75, 0.70),       # slight drop but < threshold
            ]
        )
        assert report.degraded_skills == ["coding"]

    def test_str_output_contains_table_headers(self) -> None:
        report = self._make_report([("reasoning", 0.8, 0.7)])
        output = str(report)
        assert "Before" in output
        assert "After" in output

    def test_str_output_contains_snapshot_names(self) -> None:
        report = ForgettingReport(
            snapshot_before="my_before",
            snapshot_after="my_after",
            threshold=0.10,
            comparisons=[
                CategoryComparison(category="coding", score_before=0.8, score_after=0.9)
            ],
        )
        output = str(report)
        assert "my_before" in output
        assert "my_after" in output


# ── ForgettingDetector ────────────────────────────────────────────────────────


class TestForgettingDetector:
    def test_default_threshold(self) -> None:
        d = ForgettingDetector()
        assert d.threshold == 0.10

    def test_custom_threshold(self) -> None:
        d = ForgettingDetector(threshold=0.05)
        assert d.threshold == 0.05

    def test_compare_returns_report(self) -> None:
        detector = ForgettingDetector()
        before = _make_snapshot("before", {"reasoning": 0.8, "coding": 0.75})
        after = _make_snapshot("after", {"reasoning": 0.8, "coding": 0.75})
        report = detector.compare(before, after)
        assert isinstance(report, ForgettingReport)

    def test_compare_correct_snapshot_names(self) -> None:
        detector = ForgettingDetector()
        before = _make_snapshot("snap_a", {"reasoning": 0.8})
        after = _make_snapshot("snap_b", {"reasoning": 0.7})
        report = detector.compare(before, after)
        assert report.snapshot_before == "snap_a"
        assert report.snapshot_after == "snap_b"

    def test_compare_all_categories_present(self) -> None:
        detector = ForgettingDetector()
        before = _make_snapshot("b", {"reasoning": 0.8, "coding": 0.7, "safety": 0.9})
        after = _make_snapshot("a", {"reasoning": 0.8, "coding": 0.5, "safety": 0.9})
        report = detector.compare(before, after)
        categories = [c.category for c in report.comparisons]
        assert set(categories) == {"reasoning", "coding", "safety"}

    def test_detects_forgetting_across_compare(self) -> None:
        detector = ForgettingDetector(threshold=0.10)
        before = _make_snapshot("b", {"coding": 0.85})
        after = _make_snapshot("a", {"coding": 0.60})
        report = detector.compare(before, after)
        assert "coding" in report.degraded_skills

    def test_no_false_positive_on_equal_snapshots(self) -> None:
        detector = ForgettingDetector()
        snap = _make_snapshot("same", {"reasoning": 0.8, "coding": 0.75})
        report = detector.compare(snap, snap)
        assert report.is_healthy is True

    def test_missing_category_in_after_treated_as_zero(self) -> None:
        detector = ForgettingDetector(threshold=0.05)
        before = _make_snapshot("b", {"reasoning": 0.8, "new_skill": 0.9})
        after = _make_snapshot("a", {"reasoning": 0.8})  # new_skill missing
        report = detector.compare(before, after)
        new_skill_comp = next(c for c in report.comparisons if c.category == "new_skill")
        assert new_skill_comp.score_after == 0.0
