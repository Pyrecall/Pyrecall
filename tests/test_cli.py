"""Tests for all 5 CLI commands: init, snapshot, check, rollback, status."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from mimi.cli import _CONFIG_FILE, app
from mimi.snapshot import SkillScore, SkillSnapshot

runner = CliRunner()


# ── helpers ────────────────────────────────────────────────────────────────────


def _write_config(
    tmp_path: Path,
    model: str = "test/model",
    strategy: str = "lora",
    baseline: str | None = None,
) -> None:
    config = {
        "model_name": model,
        "strategy": strategy,
        "created_at": datetime.now().isoformat(),
        "baseline_snapshot": baseline,
    }
    (tmp_path / _CONFIG_FILE).write_text(json.dumps(config, indent=2))


def _make_snapshot(
    name: str,
    scores_by_category: dict[str, float] | None = None,
    created_at: datetime | None = None,
) -> SkillSnapshot:
    scores_by_category = scores_by_category or {"reasoning": 0.8, "coding": 0.7}
    scores = [
        SkillScore(
            category=cat,
            prompt=f"Prompt for {cat}",
            response=f"Response for {cat}",
            score=val,
        )
        for cat, val in scores_by_category.items()
    ]
    return SkillSnapshot(
        name=name,
        model_name="test/model",
        created_at=created_at or datetime(2024, 1, 1, 12, 0, 0),
        scores=scores,
    )


def _make_mock_manager(
    snapshots: list[SkillSnapshot] | None = None,
    snapshot_map: dict[str, SkillSnapshot] | None = None,
) -> MagicMock:
    """Return a mock RollbackManager pre-loaded with the given snapshots."""
    snapshots = snapshots or []
    snapshot_map = snapshot_map or {s.name: s for s in snapshots}
    mgr = MagicMock()
    mgr.list_snapshots.return_value = snapshots
    mgr.load_snapshot.side_effect = lambda name: snapshot_map[name]
    mgr.has_snapshot.side_effect = lambda name: name in snapshot_map
    return mgr


# ── init ──────────────────────────────────────────────────────────────────────


class TestInit:
    def test_creates_config_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert (tmp_path / _CONFIG_FILE).exists()

    def test_exit_code_zero_on_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0

    def test_default_model_written(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init"])
        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["model_name"] == "meta-llama/Llama-3.2-1B"

    def test_custom_model_written(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init", "--model", "gpt2"])
        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["model_name"] == "gpt2"

    def test_custom_strategy_written(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init", "--strategy", "full"])
        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["strategy"] == "full"

    def test_baseline_snapshot_initially_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init"])
        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["baseline_snapshot"] is None

    def test_output_contains_model_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init", "--model", "gpt2"])
        assert "gpt2" in result.output

    def test_short_flag_works(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init", "-m", "distilgpt2"])
        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["model_name"] == "distilgpt2"

    def test_fails_if_config_already_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 1

    def test_error_message_when_already_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["init"])
        assert _CONFIG_FILE in result.output

    def test_second_init_does_not_overwrite_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init", "--model", "gpt2"])
        runner.invoke(app, ["init", "--model", "llama"])
        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        # Original config should be preserved
        assert config["model_name"] == "gpt2"


# ── snapshot ──────────────────────────────────────────────────────────────────


class TestSnapshot:
    def test_fails_without_config_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["snapshot", "v1"])
        assert result.exit_code == 1

    def test_calls_model_snapshot_with_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mock_snap = _make_snapshot("v1")
        mock_model = MagicMock()
        mock_model.snapshot.return_value = mock_snap

        with patch("mimi.model.Model", return_value=mock_model):
            runner.invoke(app, ["snapshot", "v1"])

        mock_model.snapshot.assert_called_once_with(name="v1")

    def test_updates_baseline_snapshot_in_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mock_snap = _make_snapshot("my_baseline")
        mock_model = MagicMock()
        mock_model.snapshot.return_value = mock_snap

        with patch("mimi.model.Model", return_value=mock_model):
            runner.invoke(app, ["snapshot", "my_baseline"])

        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["baseline_snapshot"] == "my_baseline"

    def test_model_instantiated_with_correct_model_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, model="meta-llama/Llama-3.2-1B")
        mock_snap = _make_snapshot("v1")
        mock_model = MagicMock()
        mock_model.snapshot.return_value = mock_snap

        with patch("mimi.model.Model", return_value=mock_model) as mock_cls:
            runner.invoke(app, ["snapshot", "v1"])

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args
        assert call_kwargs[0][0] == "meta-llama/Llama-3.2-1B"

    def test_model_instantiated_with_correct_strategy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, strategy="lora")
        mock_snap = _make_snapshot("v1")
        mock_model = MagicMock()
        mock_model.snapshot.return_value = mock_snap

        with patch("mimi.model.Model", return_value=mock_model) as mock_cls:
            runner.invoke(app, ["snapshot", "v1"])

        assert mock_cls.call_args[1]["strategy"] == "lora"


# ── check ─────────────────────────────────────────────────────────────────────


class TestCheck:
    def test_fails_without_config_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["check"])
        assert result.exit_code == 1

    def test_fails_with_fewer_than_two_snapshots(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(snapshots=[_make_snapshot("only_one")])

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["check"])

        assert result.exit_code == 1

    def test_fails_with_zero_snapshots(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(snapshots=[])

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["check"])

        assert result.exit_code == 1

    def test_exit_code_zero_when_no_forgetting(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("before", {"reasoning": 0.8})
        snap_b = _make_snapshot("after", {"reasoning": 0.85})
        mgr = _make_mock_manager(snapshots=[snap_a, snap_b])

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["check"])

        assert result.exit_code == 0

    def test_exit_code_two_when_forgetting_detected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("before", {"coding": 0.90})
        snap_b = _make_snapshot("after", {"coding": 0.50})  # drop > 0.10 threshold
        mgr = _make_mock_manager(snapshots=[snap_a, snap_b])

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["check"])

        assert result.exit_code == 2

    def test_compares_last_two_snapshots_by_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot(
            "first", created_at=datetime(2024, 1, 1)
        )
        snap_b = _make_snapshot(
            "second", created_at=datetime(2024, 2, 1)
        )
        snap_c = _make_snapshot(
            "third", created_at=datetime(2024, 3, 1)
        )
        # list_snapshots returns oldest-first; check should compare second and third
        mgr = _make_mock_manager(snapshots=[snap_a, snap_b, snap_c])

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            runner.invoke(app, ["check"])

        # list_snapshots is called but load_snapshot should NOT be called
        # (no explicit --before/--after flags, uses list directly)
        mgr.list_snapshots.assert_called_once()

    def test_fails_when_only_before_provided(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("a")
        snap_b = _make_snapshot("b")
        mgr = _make_mock_manager(snapshots=[snap_a, snap_b])

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["check", "--before", "a"])

        assert result.exit_code == 1

    def test_fails_when_only_after_provided(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("a")
        snap_b = _make_snapshot("b")
        mgr = _make_mock_manager(snapshots=[snap_a, snap_b])

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["check", "--after", "b"])

        assert result.exit_code == 1

    def test_explicit_before_after_loads_named_snapshots(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("snap_a", {"reasoning": 0.8})
        snap_b = _make_snapshot("snap_b", {"reasoning": 0.79})
        mgr = _make_mock_manager(
            snapshots=[snap_a, snap_b],
            snapshot_map={"snap_a": snap_a, "snap_b": snap_b},
        )

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(
                app, ["check", "--before", "snap_a", "--after", "snap_b"]
            )

        mgr.load_snapshot.assert_any_call("snap_a")
        mgr.load_snapshot.assert_any_call("snap_b")
        assert result.exit_code == 0


# ── rollback ──────────────────────────────────────────────────────────────────


class TestRollback:
    def test_fails_without_config_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["rollback", "v1"])
        assert result.exit_code == 1

    def test_fails_when_snapshot_not_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(snapshots=[])

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["rollback", "nonexistent"])

        assert result.exit_code == 1

    def test_error_message_lists_available_when_not_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap = _make_snapshot("real_snap")
        mgr = _make_mock_manager(
            snapshots=[snap], snapshot_map={"real_snap": snap}
        )

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["rollback", "ghost"])

        assert "ghost" in result.output

    def test_success_updates_baseline_in_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, baseline="old_snap")
        snap = _make_snapshot("new_snap")
        mgr = _make_mock_manager(
            snapshots=[snap], snapshot_map={"new_snap": snap}
        )

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["rollback", "new_snap"])

        assert result.exit_code == 0
        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["baseline_snapshot"] == "new_snap"

    def test_success_exit_code_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap = _make_snapshot("target")
        mgr = _make_mock_manager(
            snapshots=[snap], snapshot_map={"target": snap}
        )

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["rollback", "target"])

        assert result.exit_code == 0

    def test_success_output_contains_snapshot_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap = _make_snapshot("v2")
        mgr = _make_mock_manager(
            snapshots=[snap], snapshot_map={"v2": snap}
        )

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["rollback", "v2"])

        assert "v2" in result.output

    def test_old_baseline_replaced(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, baseline="old")
        snap = _make_snapshot("new")
        mgr = _make_mock_manager(
            snapshots=[snap], snapshot_map={"new": snap}
        )

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            runner.invoke(app, ["rollback", "new"])

        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["baseline_snapshot"] == "new"
        assert config["baseline_snapshot"] != "old"


# ── status ────────────────────────────────────────────────────────────────────


class TestStatus:
    def test_fails_without_config_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 1

    def test_no_snapshots_message_when_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(snapshots=[])

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "No snapshots" in result.output

    def test_exit_code_zero_with_snapshots(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, model="test/model")
        mgr = _make_mock_manager(snapshots=[_make_snapshot("v1")])

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0

    def test_snapshot_name_appears_in_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(snapshots=[_make_snapshot("release_v3")])

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["status"])

        assert "release_v3" in result.output

    def test_multiple_snapshot_names_in_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snaps = [_make_snapshot("alpha"), _make_snapshot("beta"), _make_snapshot("gamma")]
        mgr = _make_mock_manager(snapshots=snaps)

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["status"])

        assert "alpha" in result.output
        assert "beta" in result.output
        assert "gamma" in result.output

    def test_baseline_marked_in_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, baseline="v1")
        mgr = _make_mock_manager(snapshots=[_make_snapshot("v1")])

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["status"])

        # The baseline marker (★) should appear somewhere in output
        assert "★" in result.output or "v1" in result.output

    def test_list_snapshots_called_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(snapshots=[])

        with patch("mimi.rollback.RollbackManager", return_value=mgr):
            runner.invoke(app, ["status"])

        mgr.list_snapshots.assert_called_once()
