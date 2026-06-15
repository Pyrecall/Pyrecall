"""Tests for ForgettingDetector and ForgettingReport."""

from __future__ import annotations

from datetime import datetime

import pytest

from pyrecall.detector import (
    CategoryComparison,
    ForgettingDetector,
    ForgettingReport,
    PromptComparison,
)
from pyrecall.snapshot import SkillScore, SkillSnapshot

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
        report = self._make_report([("reasoning", 0.8, 0.85), ("coding", 0.7, 0.72)])
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
                ("reasoning", 0.80, 0.85),  # improved
                ("coding", 0.80, 0.60),  # forgotten
                ("safety", 0.75, 0.70),  # slight drop but < threshold
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
            comparisons=[CategoryComparison(category="coding", score_before=0.8, score_after=0.9)],
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


class TestForgettingReportSerialization:
    def test_to_dict_returns_dict(self) -> None:
        detector = ForgettingDetector(threshold=0.10)
        before = _make_snapshot("before", {"coding": 0.8, "reasoning": 0.75})
        after = _make_snapshot("after", {"coding": 0.65, "reasoning": 0.76})
        report = detector.compare(before, after)
        result = report.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_top_level_keys(self) -> None:
        detector = ForgettingDetector(threshold=0.10)
        before = _make_snapshot("b", {"coding": 0.8})
        after = _make_snapshot("a", {"coding": 0.79})
        report = detector.compare(before, after)
        d = report.to_dict()
        for key in (
            "is_healthy",
            "snapshot_before",
            "snapshot_after",
            "threshold",
            "degraded_skills",
            "comparisons",
        ):
            assert key in d

    def test_to_dict_healthy_reflects_report(self) -> None:
        detector = ForgettingDetector(threshold=0.10)
        before = _make_snapshot("b", {"coding": 0.9})
        after = _make_snapshot("a", {"coding": 0.5})
        report = detector.compare(before, after)
        assert report.to_dict()["is_healthy"] is False

    def test_to_dict_degraded_skills_populated(self) -> None:
        detector = ForgettingDetector(threshold=0.10)
        before = _make_snapshot("b", {"coding": 0.9, "safety": 0.85})
        after = _make_snapshot("a", {"coding": 0.5, "safety": 0.84})
        report = detector.compare(before, after)
        d = report.to_dict()
        assert "coding" in d["degraded_skills"]
        assert "safety" not in d["degraded_skills"]

    def test_to_dict_comparison_fields(self) -> None:
        detector = ForgettingDetector(threshold=0.10)
        before = _make_snapshot("b", {"coding": 0.8})
        after = _make_snapshot("a", {"coding": 0.7})
        report = detector.compare(before, after)
        comp = report.to_dict()["comparisons"][0]
        assert comp["category"] == "coding"
        assert comp["score_before"] == pytest.approx(0.8, abs=0.001)
        assert comp["score_after"] == pytest.approx(0.7, abs=0.001)
        assert comp["delta"] == pytest.approx(-0.1, abs=0.001)
        assert "pct_change" in comp
        assert comp["status"] in ("OK", "FORGOTTEN")

    def test_to_dict_status_forgotten_when_degraded(self) -> None:
        detector = ForgettingDetector(threshold=0.10)
        before = _make_snapshot("b", {"coding": 0.9})
        after = _make_snapshot("a", {"coding": 0.5})
        report = detector.compare(before, after)
        comp = report.to_dict()["comparisons"][0]
        assert comp["status"] == "FORGOTTEN"

    def test_to_dict_status_ok_when_healthy(self) -> None:
        detector = ForgettingDetector(threshold=0.10)
        before = _make_snapshot("b", {"coding": 0.8})
        after = _make_snapshot("a", {"coding": 0.79})
        report = detector.compare(before, after)
        comp = report.to_dict()["comparisons"][0]
        assert comp["status"] == "OK"

    def test_to_json_is_valid_json(self) -> None:
        import json

        detector = ForgettingDetector(threshold=0.10)
        before = _make_snapshot("b", {"coding": 0.8})
        after = _make_snapshot("a", {"coding": 0.75})
        report = detector.compare(before, after)
        parsed = json.loads(report.to_json())
        assert parsed["snapshot_before"] == "b"
        assert parsed["snapshot_after"] == "a"

    def test_to_dict_comparisons_include_prompts_key(self) -> None:
        detector = ForgettingDetector(threshold=0.10)
        before = _make_snapshot("b", {"coding": 0.8})
        after = _make_snapshot("a", {"coding": 0.6})
        report = detector.compare(before, after)
        comp = report.to_dict()["comparisons"][0]
        assert "prompts" in comp
        assert len(comp["prompts"]) == 1

    def test_to_dict_prompt_entry_fields(self) -> None:
        detector = ForgettingDetector(threshold=0.10)
        before = _make_snapshot("b", {"coding": 0.8})
        after = _make_snapshot("a", {"coding": 0.6})
        report = detector.compare(before, after)
        p = report.to_dict()["comparisons"][0]["prompts"][0]
        for key in ("category", "prompt", "score_before", "score_after", "delta"):
            assert key in p


class TestPromptComparisons:
    def _make_multi_prompt_snapshot(
        self, name: str, cat_prompts: dict[str, list[float]]
    ) -> SkillSnapshot:
        scores = []
        for cat, vals in cat_prompts.items():
            for i, v in enumerate(vals):
                scores.append(
                    SkillScore(category=cat, prompt=f"prompt_{cat}_{i}", response="r", score=v)
                )
        return SkillSnapshot(
            name=name, model_name="m", created_at=datetime(2024, 1, 1), scores=scores
        )

    def test_compare_populates_prompt_comparisons(self) -> None:
        before = self._make_multi_prompt_snapshot("b", {"coding": [0.8, 0.7]})
        after = self._make_multi_prompt_snapshot("a", {"coding": [0.6, 0.5]})
        report = ForgettingDetector().compare(before, after)
        assert len(report.prompt_comparisons) == 2

    def test_prompts_for_category_sorted_worst_first(self) -> None:
        before = self._make_multi_prompt_snapshot("b", {"coding": [0.9, 0.8]})
        after = self._make_multi_prompt_snapshot("a", {"coding": [0.4, 0.75]})
        report = ForgettingDetector().compare(before, after)
        prompts = report.prompts_for_category("coding")
        assert prompts[0].delta < prompts[1].delta

    def test_prompts_for_category_filters_correctly(self) -> None:
        before = self._make_multi_prompt_snapshot("b", {"coding": [0.8], "safety": [0.9]})
        after = self._make_multi_prompt_snapshot("a", {"coding": [0.7], "safety": [0.8]})
        report = ForgettingDetector().compare(before, after)
        assert all(p.category == "coding" for p in report.prompts_for_category("coding"))

    def test_prompt_comparison_delta(self) -> None:
        p = PromptComparison(category="c", prompt="q", score_before=0.8, score_after=0.6)
        assert p.delta == pytest.approx(-0.2, abs=1e-6)

    def test_verbose_render_includes_prompt_text(self) -> None:
        before = self._make_multi_prompt_snapshot("b", {"coding": [0.9]})
        after = self._make_multi_prompt_snapshot("a", {"coding": [0.5]})
        report = ForgettingDetector(threshold=0.10).compare(before, after)
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        report._render(Console(file=buf, highlight=False), verbose=True)
        output = buf.getvalue()
        assert "prompt_coding_0" in output

    def test_non_verbose_render_omits_prompt_table(self) -> None:
        before = self._make_multi_prompt_snapshot("b", {"coding": [0.9]})
        after = self._make_multi_prompt_snapshot("a", {"coding": [0.5]})
        report = ForgettingDetector(threshold=0.10).compare(before, after)
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        report._render(Console(file=buf, highlight=False), verbose=False)
        output = buf.getvalue()
        assert "prompt_coding_0" not in output


# ── per-category thresholds ───────────────────────────────────────────────────


class TestPerCategoryThresholds:
    def test_category_threshold_overrides_global(self) -> None:
        detector = ForgettingDetector(threshold=0.10, category_thresholds={"safety": 0.03})
        before = _make_snapshot("before", {"safety": 0.90, "coding": 0.80})
        after = _make_snapshot("after", {"safety": 0.86, "coding": 0.79})
        report = detector.compare(before, after)
        # safety dropped 0.04 — over the 0.03 override but under the 0.10 global
        assert "safety" in report.degraded_skills
        assert "coding" not in report.degraded_skills

    def test_global_threshold_used_when_no_override(self) -> None:
        detector = ForgettingDetector(threshold=0.10, category_thresholds={"safety": 0.03})
        before = _make_snapshot("before", {"coding": 0.80})
        after = _make_snapshot("after", {"coding": 0.77})
        report = detector.compare(before, after)
        # coding dropped 0.03 — under the 0.10 global, no override for coding
        assert "coding" not in report.degraded_skills

    def test_threshold_for_returns_override(self) -> None:
        detector = ForgettingDetector(threshold=0.10, category_thresholds={"safety": 0.03})
        before = _make_snapshot("before", {"safety": 0.90})
        after = _make_snapshot("after", {"safety": 0.90})
        report = detector.compare(before, after)
        assert report._threshold_for("safety") == pytest.approx(0.03)

    def test_threshold_for_returns_global_for_unknown_category(self) -> None:
        detector = ForgettingDetector(threshold=0.10, category_thresholds={"safety": 0.03})
        before = _make_snapshot("before", {"coding": 0.80})
        after = _make_snapshot("after", {"coding": 0.80})
        report = detector.compare(before, after)
        assert report._threshold_for("coding") == pytest.approx(0.10)

    def test_to_dict_includes_per_category_threshold(self) -> None:
        detector = ForgettingDetector(threshold=0.10, category_thresholds={"safety": 0.03})
        before = _make_snapshot("before", {"safety": 0.90, "coding": 0.80})
        after = _make_snapshot("after", {"safety": 0.90, "coding": 0.80})
        report = detector.compare(before, after)
        data = report.to_dict()
        thresholds = {c["category"]: c["threshold"] for c in data["comparisons"]}
        assert thresholds["safety"] == pytest.approx(0.03)
        assert thresholds["coding"] == pytest.approx(0.10)

    def test_no_category_thresholds_uses_global_for_all(self) -> None:
        detector = ForgettingDetector(threshold=0.10)
        before = _make_snapshot("before", {"coding": 0.80, "safety": 0.90})
        after = _make_snapshot("after", {"coding": 0.80, "safety": 0.90})
        report = detector.compare(before, after)
        for comp in report.comparisons:
            assert report._threshold_for(comp.category) == pytest.approx(0.10)

    def test_category_thresholds_stored_on_report(self) -> None:
        cat_thresh = {"safety": 0.03, "coding": 0.15}
        detector = ForgettingDetector(threshold=0.10, category_thresholds=cat_thresh)
        before = _make_snapshot("before", {"coding": 0.80})
        after = _make_snapshot("after", {"coding": 0.80})
        report = detector.compare(before, after)
        assert report.category_thresholds == cat_thresh


# ── Cohen's d and severity levels ────────────────────────────────────────────


def _make_snapshot_with_items(name: str, scores_per_cat: dict[str, list[float]]) -> SkillSnapshot:
    """Build a SkillSnapshot with multiple SkillScore items per category."""
    skill_scores = [
        SkillScore(
            category=cat,
            prompt=f"Prompt {cat} {i}",
            response=f"Response {cat} {i}",
            score=score,
        )
        for cat, vals in scores_per_cat.items()
        for i, score in enumerate(vals)
    ]
    return SkillSnapshot(
        name=name,
        model_name="test/model",
        created_at=datetime(2024, 1, 1),
        scores=skill_scores,
    )


class TestCohensD:
    def test_cohen_d_zero_when_no_change(self) -> None:
        scores = [0.8, 0.75, 0.82, 0.78, 0.80]
        before = _make_snapshot_with_items("b", {"coding": scores})
        after = _make_snapshot_with_items("a", {"coding": scores})
        report = ForgettingDetector().compare(before, after)
        comp = next(c for c in report.comparisons if c.category == "coding")
        assert comp.cohen_d == pytest.approx(0.0)

    def test_cohen_d_negative_when_scores_drop(self) -> None:
        before = _make_snapshot_with_items("b", {"coding": [0.9, 0.85, 0.88, 0.91, 0.87]})
        after = _make_snapshot_with_items("a", {"coding": [0.7, 0.65, 0.68, 0.71, 0.67]})
        report = ForgettingDetector().compare(before, after)
        comp = next(c for c in report.comparisons if c.category == "coding")
        assert comp.cohen_d < 0.0

    def test_cohen_d_positive_when_scores_improve(self) -> None:
        before = _make_snapshot_with_items("b", {"coding": [0.6, 0.62, 0.58, 0.61, 0.59]})
        after = _make_snapshot_with_items("a", {"coding": [0.9, 0.88, 0.91, 0.87, 0.89]})
        report = ForgettingDetector().compare(before, after)
        comp = next(c for c in report.comparisons if c.category == "coding")
        assert comp.cohen_d > 0.0

    def test_n_items_stored_on_comparison(self) -> None:
        before = _make_snapshot_with_items("b", {"coding": [0.8, 0.82, 0.79, 0.81, 0.80]})
        after = _make_snapshot_with_items("a", {"coding": [0.78, 0.80, 0.77, 0.79, 0.78]})
        report = ForgettingDetector().compare(before, after)
        comp = next(c for c in report.comparisons if c.category == "coding")
        assert comp.n_items == 5

    def test_cohen_d_zero_when_single_item(self) -> None:
        before = _make_snapshot_with_items("b", {"coding": [0.8]})
        after = _make_snapshot_with_items("a", {"coding": [0.6]})
        report = ForgettingDetector().compare(before, after)
        comp = next(c for c in report.comparisons if c.category == "coding")
        assert comp.cohen_d == pytest.approx(0.0)


class TestSeverityLevels:
    def test_ok_when_no_drop(self) -> None:
        c = CategoryComparison(
            category="coding", score_before=0.8, score_after=0.8, cohen_d=0.0, n_items=5
        )
        assert c.severity == "OK"

    def test_ok_when_improvement(self) -> None:
        c = CategoryComparison(
            category="coding", score_before=0.7, score_after=0.85, cohen_d=1.2, n_items=5
        )
        assert c.severity == "OK"

    def test_minor_when_small_effect(self) -> None:
        c = CategoryComparison(
            category="coding", score_before=0.8, score_after=0.79, cohen_d=-0.15, n_items=5
        )
        assert c.severity == "MINOR"

    def test_moderate_when_medium_effect(self) -> None:
        c = CategoryComparison(
            category="coding", score_before=0.8, score_after=0.74, cohen_d=-0.35, n_items=10
        )
        assert c.severity == "MODERATE"

    def test_severe_when_large_medium_effect(self) -> None:
        c = CategoryComparison(
            category="coding", score_before=0.8, score_after=0.65, cohen_d=-0.65, n_items=10
        )
        assert c.severity == "SEVERE"

    def test_critical_when_very_large_effect(self) -> None:
        c = CategoryComparison(
            category="coding", score_before=0.8, score_after=0.50, cohen_d=-1.1, n_items=10
        )
        assert c.severity == "CRITICAL"

    def test_severity_in_to_dict(self) -> None:
        before = _make_snapshot_with_items("b", {"coding": [0.9] * 10})
        after = _make_snapshot_with_items("a", {"coding": [0.5] * 10})
        report = ForgettingDetector().compare(before, after)
        comp_dict = next(c for c in report.to_dict()["comparisons"] if c["category"] == "coding")
        assert "severity" in comp_dict
        assert comp_dict["severity"] in ("OK", "MINOR", "MODERATE", "SEVERE", "CRITICAL")

    def test_cohen_d_in_to_dict(self) -> None:
        before = _make_snapshot_with_items("b", {"coding": [0.8] * 5})
        after = _make_snapshot_with_items("a", {"coding": [0.8] * 5})
        report = ForgettingDetector().compare(before, after)
        comp_dict = next(c for c in report.to_dict()["comparisons"] if c["category"] == "coding")
        assert "cohen_d" in comp_dict
        assert isinstance(comp_dict["cohen_d"], float)


class TestSingleItemSeverityFallback:
    """Severity should use delta-based buckets when n_items < 2 (Cohen's d unavailable)."""

    def test_severity_ok_when_no_drop_single_item(self) -> None:
        c = CategoryComparison(
            category="qa", score_before=0.8, score_after=0.85, cohen_d=0.0, n_items=1
        )
        assert c.severity == "OK"

    def test_severity_minor_for_tiny_drop_single_item(self) -> None:
        c = CategoryComparison(
            category="qa", score_before=0.8, score_after=0.77, cohen_d=0.0, n_items=1
        )
        assert c.severity == "MINOR"

    def test_severity_moderate_single_item(self) -> None:
        c = CategoryComparison(
            category="qa", score_before=0.8, score_after=0.70, cohen_d=0.0, n_items=1
        )
        assert c.severity == "MODERATE"

    def test_severity_severe_single_item(self) -> None:
        c = CategoryComparison(
            category="qa", score_before=0.8, score_after=0.62, cohen_d=0.0, n_items=1
        )
        assert c.severity == "SEVERE"

    def test_severity_critical_single_item(self) -> None:
        c = CategoryComparison(
            category="qa", score_before=0.8, score_after=0.40, cohen_d=0.0, n_items=1
        )
        assert c.severity == "CRITICAL"

    def test_severity_critical_not_minor_for_large_drop(self) -> None:
        """Regression: single-item 0.4-point drop must NOT return MINOR."""
        c = CategoryComparison(
            category="domain_qa", score_before=0.9, score_after=0.5, cohen_d=0.0, n_items=1
        )
        assert c.severity != "MINOR"
        assert c.severity == "CRITICAL"

    def test_threshold_based_severity_property(self) -> None:
        c = CategoryComparison(
            category="qa", score_before=0.9, score_after=0.5, cohen_d=0.0, n_items=1
        )
        assert c.threshold_based_severity == "CRITICAL"

    def test_severity_uses_cohen_d_when_n_items_ge_2(self) -> None:
        # Large drop but tiny non-zero Cohen's d → should be MINOR via effect-size path.
        c = CategoryComparison(
            category="qa", score_before=0.8, score_after=0.40, cohen_d=-0.1, n_items=5
        )
        assert c.severity == "MINOR"

    def test_severity_zero_variance_multi_item_falls_back_to_delta(self) -> None:
        # cohen_d=0 with n_items>=2 and real drop means std_d was zero (identical deltas).
        # Must use threshold_based_severity, not return MINOR.
        c = CategoryComparison(
            category="safety", score_before=0.9, score_after=0.1, cohen_d=0.0, n_items=5
        )
        assert c.severity == "CRITICAL"

    def test_severity_zero_variance_no_drop_is_ok(self) -> None:
        # cohen_d=0 but delta>=0 → OK, not a fallback case.
        c = CategoryComparison(
            category="safety", score_before=0.8, score_after=0.8, cohen_d=0.0, n_items=5
        )
        assert c.severity == "OK"

    def test_render_footnote_present_for_single_item(self) -> None:
        from io import StringIO

        from rich.console import Console

        report = ForgettingReport(
            snapshot_before="b",
            snapshot_after="a",
            threshold=0.10,
            comparisons=[
                CategoryComparison(
                    category="domain_qa", score_before=0.9, score_after=0.5, cohen_d=0.0, n_items=1
                )
            ],
        )
        buf = StringIO()
        report._render(Console(file=buf, highlight=False))
        assert "Cohen's d requires" in buf.getvalue()

    def test_render_no_footnote_when_all_multi_item(self) -> None:
        from io import StringIO

        from rich.console import Console

        report = ForgettingReport(
            snapshot_before="b",
            snapshot_after="a",
            threshold=0.10,
            comparisons=[
                CategoryComparison(
                    category="coding", score_before=0.8, score_after=0.7, cohen_d=-0.5, n_items=5
                )
            ],
        )
        buf = StringIO()
        report._render(Console(file=buf, highlight=False))
        assert "Cohen's d requires" not in buf.getvalue()

    def test_compare_single_prompt_category_severity_not_minor_for_large_drop(self) -> None:
        """End-to-end: custom suite with 1 prompt per category, big drop → not MINOR."""
        detector = ForgettingDetector(threshold=0.10)
        before = _make_snapshot("before", {"domain_qa": 0.9})
        after = _make_snapshot("after", {"domain_qa": 0.5})
        report = detector.compare(before, after)
        comp = next(c for c in report.comparisons if c.category == "domain_qa")
        assert comp.n_items == 1
        assert comp.severity not in ("MINOR", "OK")


class TestSeverityMethodField:
    def test_severity_method_effect_size_when_multi_item(self) -> None:
        c = CategoryComparison(
            category="coding", score_before=0.8, score_after=0.7, cohen_d=-0.5, n_items=5
        )
        assert c.severity_method == "effect_size"

    def test_severity_method_delta_when_single_item(self) -> None:
        c = CategoryComparison(
            category="qa", score_before=0.8, score_after=0.5, cohen_d=0.0, n_items=1
        )
        assert c.severity_method == "delta"

    def test_severity_method_delta_when_zero_items(self) -> None:
        c = CategoryComparison(category="qa", score_before=0.8, score_after=0.5, n_items=0)
        assert c.severity_method == "delta"

    def test_severity_method_in_to_dict(self) -> None:
        detector = ForgettingDetector(threshold=0.10)
        before = _make_snapshot("b", {"coding": 0.8})
        after = _make_snapshot("a", {"coding": 0.6})
        report = detector.compare(before, after)
        comp_dict = report.to_dict()["comparisons"][0]
        assert "severity_method" in comp_dict
        assert comp_dict["severity_method"] in ("effect_size", "delta")

    def test_severity_method_effect_size_in_to_dict_for_multi_prompt(self) -> None:
        # Use varied deltas so std_d > 0 and cohen_d is non-zero → effect_size path.
        before = _make_snapshot_with_items("b", {"coding": [0.8, 0.7]})
        after = _make_snapshot_with_items("a", {"coding": [0.6, 0.4]})
        report = ForgettingDetector().compare(before, after)
        comp_dict = next(c for c in report.to_dict()["comparisons"] if c["category"] == "coding")
        assert comp_dict["severity_method"] == "effect_size"

    def test_severity_method_delta_when_zero_variance_multi_prompt(self) -> None:
        # Identical deltas → zero variance → cohen_d=0 → falls back to delta method.
        before = _make_snapshot_with_items("b", {"coding": [0.9, 0.9]})
        after = _make_snapshot_with_items("a", {"coding": [0.1, 0.1]})
        report = ForgettingDetector().compare(before, after)
        comp_dict = next(c for c in report.to_dict()["comparisons"] if c["category"] == "coding")
        assert comp_dict["severity_method"] == "delta"
        assert comp_dict["severity"] == "CRITICAL"


class TestDeltaThresholdConstants:
    def test_constants_exported(self) -> None:
        from pyrecall.detector import _DELTA_MINOR, _DELTA_MODERATE, _DELTA_SEVERE

        assert _DELTA_MINOR == 0.05
        assert _DELTA_MODERATE == 0.15
        assert _DELTA_SEVERE == 0.30

    def test_constants_ordered(self) -> None:
        from pyrecall.detector import _DELTA_MINOR, _DELTA_MODERATE, _DELTA_SEVERE

        assert _DELTA_MINOR < _DELTA_MODERATE < _DELTA_SEVERE


class TestNaNToDict:
    def test_prompt_comparison_to_dict_nan_score_before_is_none(self) -> None:
        import json

        pc = PromptComparison(
            category="safety", prompt="p", score_before=float("nan"), score_after=0.7
        )
        d = pc.to_dict()
        assert d["score_before"] is None
        json.dumps(d)  # must not raise

    def test_prompt_comparison_to_dict_nan_score_after_is_none(self) -> None:
        import json

        pc = PromptComparison(
            category="safety", prompt="p", score_before=0.8, score_after=float("nan")
        )
        d = pc.to_dict()
        assert d["score_after"] is None
        json.dumps(d)  # must not raise

    def test_forgetting_report_to_json_with_nan_is_valid(self) -> None:
        import json

        before = _make_snapshot("v1", {"safety": 0.8})
        after = _make_snapshot("v2", {"safety": float("nan")})
        report = ForgettingDetector().compare(before, after)
        parsed = json.loads(report.to_json())
        comp = next(c for c in parsed["comparisons"] if c["category"] == "safety")
        assert comp["score_after"] is None

    def test_forgetting_report_to_dict_nan_delta_is_none(self) -> None:
        before = _make_snapshot("v1", {"safety": float("nan")})
        after = _make_snapshot("v2", {"safety": 0.7})
        report = ForgettingDetector().compare(before, after)
        comp = next(c for c in report.to_dict()["comparisons"] if c["category"] == "safety")
        assert comp["score_before"] is None
        assert comp["delta"] is None


class TestBenchmarkCount:
    def test_default_benchmarks_total_180(self) -> None:
        from pyrecall.benchmarks.default import DEFAULT_BENCHMARKS

        assert len(DEFAULT_BENCHMARKS) == 180

    def test_each_category_has_20_items(self) -> None:
        from collections import Counter

        from pyrecall.benchmarks.default import CATEGORIES, DEFAULT_BENCHMARKS

        counts = Counter(b.category for b in DEFAULT_BENCHMARKS)
        for cat in CATEGORIES:
            assert counts[cat] == 20, f"{cat} has {counts[cat]} items, expected 20"


class TestNaNPctChangeAndSeverityMethod:
    def test_pct_change_nan_when_score_before_nan(self) -> None:
        import math

        c = CategoryComparison(category="safety", score_before=float("nan"), score_after=0.7)
        assert math.isnan(c.pct_change)

    def test_pct_change_nan_when_score_after_nan(self) -> None:
        import math

        c = CategoryComparison(category="safety", score_before=0.8, score_after=float("nan"))
        assert math.isnan(c.pct_change)

    def test_severity_method_unknown_when_score_before_nan(self) -> None:
        c = CategoryComparison(
            category="safety", score_before=float("nan"), score_after=0.7, n_items=5
        )
        assert c.severity_method == "unknown"

    def test_severity_method_unknown_when_score_after_nan(self) -> None:
        c = CategoryComparison(
            category="safety", score_before=0.8, score_after=float("nan"), n_items=5
        )
        assert c.severity_method == "unknown"


class TestMarkdownAndHTMLEscaping:
    def test_pipe_in_category_name_escaped_in_markdown(self) -> None:
        report = ForgettingReport(
            snapshot_before="b",
            snapshot_after="a",
            threshold=0.1,
            comparisons=[CategoryComparison(category="foo|bar", score_before=0.8, score_after=0.7)],
        )
        md = report.to_markdown()
        table_lines = [l for l in md.splitlines() if l.startswith("|") and "foo" in l]
        assert len(table_lines) == 1
        assert r"foo\|bar" in table_lines[0]

    def test_html_tag_in_category_name_escaped_in_html(self) -> None:
        report = ForgettingReport(
            snapshot_before="b",
            snapshot_after="a",
            threshold=0.1,
            comparisons=[
                CategoryComparison(
                    category="<script>alert(1)</script>", score_before=0.8, score_after=0.7
                )
            ],
        )
        html = report.to_html()
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
