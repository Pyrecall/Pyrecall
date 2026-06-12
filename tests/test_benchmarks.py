"""Tests for CustomBenchmarkManager and benchmark CLI subcommands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from pyrecall.benchmarks.custom import CustomBenchmarkManager
from pyrecall.benchmarks.default import Benchmark
from pyrecall.cli import app

runner = CliRunner()


# ── helpers ────────────────────────────────────────────────────────────────────


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries))


def _valid_entry(
    prompt: str = "What is port?",
    reference_answer: str = "The left side.",
    category: str | None = None,
) -> dict:
    entry: dict = {"prompt": prompt, "reference_answer": reference_answer}
    if category is not None:
        entry["category"] = category
    return entry


# ── CustomBenchmarkManager ────────────────────────────────────────────────────


class TestCustomBenchmarkManagerAdd:
    def test_add_copies_file(self, tmp_path: Path) -> None:
        src = tmp_path / "suite.jsonl"
        _write_jsonl(src, [_valid_entry()])
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")

        mgr.add(src)

        assert (tmp_path / "store" / "suite.jsonl").exists()

    def test_add_returns_suite_name(self, tmp_path: Path) -> None:
        src = tmp_path / "nautical.jsonl"
        _write_jsonl(src, [_valid_entry()])
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")

        name = mgr.add(src)

        assert name == "nautical"

    def test_add_custom_name(self, tmp_path: Path) -> None:
        src = tmp_path / "raw.jsonl"
        _write_jsonl(src, [_valid_entry()])
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")

        name = mgr.add(src, name="my_domain")

        assert name == "my_domain"
        assert (tmp_path / "store" / "my_domain.jsonl").exists()

    def test_add_raises_file_not_found(self, tmp_path: Path) -> None:
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")
        with pytest.raises(FileNotFoundError):
            mgr.add(tmp_path / "missing.jsonl")

    def test_add_raises_value_error_for_empty_file(self, tmp_path: Path) -> None:
        src = tmp_path / "empty.jsonl"
        src.write_text("")
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")
        with pytest.raises(ValueError, match="no valid"):
            mgr.add(src)

    def test_add_skips_lines_missing_required_keys(self, tmp_path: Path) -> None:
        src = tmp_path / "mixed.jsonl"
        _write_jsonl(src, [{"prompt": "only prompt"}, _valid_entry()])
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")

        mgr.add(src)  # should not raise — one valid line is enough

    def test_add_raises_if_all_lines_invalid(self, tmp_path: Path) -> None:
        src = tmp_path / "bad.jsonl"
        _write_jsonl(src, [{"prompt": "no reference"}])
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")
        with pytest.raises(ValueError):
            mgr.add(src)


class TestCustomBenchmarkManagerList:
    def test_list_empty_when_no_suites(self, tmp_path: Path) -> None:
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")
        assert mgr.suites() == []

    def test_list_returns_registered_suites(self, tmp_path: Path) -> None:
        src = tmp_path / "suite.jsonl"
        _write_jsonl(src, [_valid_entry(), _valid_entry()])
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")
        mgr.add(src)

        result = mgr.suites()

        assert len(result) == 1
        assert result[0]["name"] == "suite"
        assert result[0]["count"] == 2

    def test_list_includes_path(self, tmp_path: Path) -> None:
        src = tmp_path / "suite.jsonl"
        _write_jsonl(src, [_valid_entry()])
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")
        mgr.add(src)

        assert "path" in mgr.suites()[0]


class TestCustomBenchmarkManagerRemove:
    def test_remove_deletes_file(self, tmp_path: Path) -> None:
        src = tmp_path / "suite.jsonl"
        _write_jsonl(src, [_valid_entry()])
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")
        mgr.add(src)

        mgr.remove("suite")

        assert not (tmp_path / "store" / "suite.jsonl").exists()

    def test_remove_raises_when_not_found(self, tmp_path: Path) -> None:
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")
        with pytest.raises(FileNotFoundError):
            mgr.remove("missing")


class TestCustomBenchmarkManagerLoadAll:
    def test_load_all_returns_benchmark_objects(self, tmp_path: Path) -> None:
        src = tmp_path / "suite.jsonl"
        _write_jsonl(src, [_valid_entry(prompt="p1", reference_answer="r1")])
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")
        mgr.add(src)

        items = mgr.load_all()

        assert len(items) == 1
        assert isinstance(items[0], Benchmark)
        assert items[0].prompt == "p1"
        assert items[0].reference_answer == "r1"

    def test_load_all_uses_category_field_when_present(self, tmp_path: Path) -> None:
        src = tmp_path / "suite.jsonl"
        _write_jsonl(src, [_valid_entry(category="nautical")])
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")
        mgr.add(src)

        items = mgr.load_all()

        assert items[0].category == "nautical"

    def test_load_all_falls_back_to_filename_for_category(self, tmp_path: Path) -> None:
        src = tmp_path / "my_domain.jsonl"
        _write_jsonl(src, [_valid_entry()])
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")
        mgr.add(src)

        items = mgr.load_all()

        assert items[0].category == "my_domain"

    def test_load_all_empty_when_no_suites(self, tmp_path: Path) -> None:
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")
        assert mgr.load_all() == []

    def test_load_all_merges_multiple_suites(self, tmp_path: Path) -> None:
        s1 = tmp_path / "a.jsonl"
        s2 = tmp_path / "b.jsonl"
        _write_jsonl(s1, [_valid_entry(), _valid_entry()])
        _write_jsonl(s2, [_valid_entry()])
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")
        mgr.add(s1)
        mgr.add(s2)

        assert len(mgr.load_all()) == 3


class TestCustomBenchmarkManagerCount:
    def test_count_zero_when_empty(self, tmp_path: Path) -> None:
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")
        assert mgr.count() == 0

    def test_count_sums_across_suites(self, tmp_path: Path) -> None:
        s1 = tmp_path / "a.jsonl"
        s2 = tmp_path / "b.jsonl"
        _write_jsonl(s1, [_valid_entry(), _valid_entry()])
        _write_jsonl(s2, [_valid_entry()])
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")
        mgr.add(s1)
        mgr.add(s2)

        assert mgr.count() == 3


# ── CLI: benchmark add ────────────────────────────────────────────────────────


class TestBenchmarkAddCli:
    def test_exit_code_zero_on_success(self, tmp_path: Path) -> None:
        src = tmp_path / "suite.jsonl"
        _write_jsonl(src, [_valid_entry()])
        store = tmp_path / "store"
        mgr = CustomBenchmarkManager(base_dir=store)

        with patch("pyrecall.benchmarks.custom.CustomBenchmarkManager", return_value=mgr):
            result = runner.invoke(app, ["benchmark", "add", str(src)])

        assert result.exit_code == 0

    def test_output_contains_suite_name(self, tmp_path: Path) -> None:
        src = tmp_path / "nautical.jsonl"
        _write_jsonl(src, [_valid_entry()])
        store = tmp_path / "store"
        mgr = CustomBenchmarkManager(base_dir=store)

        with patch("pyrecall.benchmarks.custom.CustomBenchmarkManager", return_value=mgr):
            result = runner.invoke(app, ["benchmark", "add", str(src)])

        assert "nautical" in result.output

    def test_fails_when_file_missing(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["benchmark", "add", str(tmp_path / "nope.jsonl")])
        assert result.exit_code == 1

    def test_custom_name_flag(self, tmp_path: Path) -> None:
        src = tmp_path / "raw.jsonl"
        _write_jsonl(src, [_valid_entry()])
        store = tmp_path / "store"
        mgr = CustomBenchmarkManager(base_dir=store)

        with patch("pyrecall.benchmarks.custom.CustomBenchmarkManager", return_value=mgr):
            result = runner.invoke(app, ["benchmark", "add", str(src), "--name", "custom_name"])

        assert result.exit_code == 0
        assert "custom_name" in result.output


# ── CLI: benchmark list ───────────────────────────────────────────────────────


class TestBenchmarkListCli:
    def test_empty_message_when_no_suites(self, tmp_path: Path) -> None:
        mgr = CustomBenchmarkManager(base_dir=tmp_path / "store")

        with patch("pyrecall.benchmarks.custom.CustomBenchmarkManager", return_value=mgr):
            result = runner.invoke(app, ["benchmark", "list"])

        assert result.exit_code == 0
        assert "no custom" in result.output.lower()

    def test_shows_registered_suite(self, tmp_path: Path) -> None:
        src = tmp_path / "nautical.jsonl"
        _write_jsonl(src, [_valid_entry()])
        store = tmp_path / "store"
        mgr = CustomBenchmarkManager(base_dir=store)
        mgr.add(src)

        with patch("pyrecall.benchmarks.custom.CustomBenchmarkManager", return_value=mgr):
            result = runner.invoke(app, ["benchmark", "list"])

        assert result.exit_code == 0
        assert "nautical" in result.output


# ── CLI: benchmark remove ─────────────────────────────────────────────────────


class TestBenchmarkRemoveCli:
    def test_removes_suite_with_yes_flag(self, tmp_path: Path) -> None:
        src = tmp_path / "suite.jsonl"
        _write_jsonl(src, [_valid_entry()])
        store = tmp_path / "store"
        mgr = CustomBenchmarkManager(base_dir=store)
        mgr.add(src)

        with patch("pyrecall.benchmarks.custom.CustomBenchmarkManager", return_value=mgr):
            result = runner.invoke(app, ["benchmark", "remove", "suite", "--yes"])

        assert result.exit_code == 0
        assert mgr.suites() == []

    def test_fails_when_suite_not_found(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        mgr = CustomBenchmarkManager(base_dir=store)

        with patch("pyrecall.benchmarks.custom.CustomBenchmarkManager", return_value=mgr):
            result = runner.invoke(app, ["benchmark", "remove", "missing", "--yes"])

        assert result.exit_code == 1
        assert "missing" in result.output.lower() or "not found" in result.output.lower()

    def test_output_confirms_removal(self, tmp_path: Path) -> None:
        src = tmp_path / "suite.jsonl"
        _write_jsonl(src, [_valid_entry()])
        store = tmp_path / "store"
        mgr = CustomBenchmarkManager(base_dir=store)
        mgr.add(src)

        with patch("pyrecall.benchmarks.custom.CustomBenchmarkManager", return_value=mgr):
            result = runner.invoke(app, ["benchmark", "remove", "suite", "--yes"])

        assert "suite" in result.output
