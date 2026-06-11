"""Tests for all CLI commands: init, learn, snapshot, check, rollback, status, replay, export."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pyrecall.cli import _CONFIG_FILE, app
from pyrecall.snapshot import SkillScore, SkillSnapshot

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
    def test_creates_config_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

    def test_default_model_written(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init"])
        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["model_name"] == "meta-llama/Llama-3.2-1B"

    def test_custom_model_written(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init", "--model", "gpt2"])
        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["model_name"] == "gpt2"

    def test_custom_strategy_written(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init", "--strategy", "qlora"])
        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["strategy"] == "qlora"

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

    def test_short_flag_works(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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


# ── learn ─────────────────────────────────────────────────────────────────────


class TestLearn:
    def _config(self, tmp_path: Path, **kwargs) -> None:
        _write_config(tmp_path, **kwargs)

    def test_fails_without_config_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        data = tmp_path / "train.jsonl"
        data.write_text('{"text": "hi"}\n')
        result = runner.invoke(app, ["learn", str(data)])
        assert result.exit_code == 1

    def test_fails_when_data_file_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._config(tmp_path)
        result = runner.invoke(app, ["learn", str(tmp_path / "nope.jsonl")])
        assert result.exit_code == 1

    def test_error_message_contains_missing_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._config(tmp_path)
        result = runner.invoke(app, ["learn", str(tmp_path / "nope.jsonl")])
        assert "nope.jsonl" in result.output

    def test_calls_model_learn_with_data_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._config(tmp_path)
        data = tmp_path / "train.jsonl"
        data.write_text('{"text": "hi"}\n')
        mock_model = MagicMock()

        with patch("pyrecall.model.Model", return_value=mock_model):
            runner.invoke(app, ["learn", str(data)])

        mock_model.learn.assert_called_once()
        assert mock_model.learn.call_args[0][0] == str(data)

    def test_default_epochs_is_three(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        self._config(tmp_path)
        data = tmp_path / "train.jsonl"
        data.write_text('{"text": "hi"}\n')
        mock_model = MagicMock()

        with patch("pyrecall.model.Model", return_value=mock_model):
            runner.invoke(app, ["learn", str(data)])

        assert mock_model.learn.call_args[1]["epochs"] == 3

    def test_custom_epochs_passed_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._config(tmp_path)
        data = tmp_path / "train.jsonl"
        data.write_text('{"text": "hi"}\n')
        mock_model = MagicMock()

        with patch("pyrecall.model.Model", return_value=mock_model):
            runner.invoke(app, ["learn", str(data), "--epochs", "7"])

        assert mock_model.learn.call_args[1]["epochs"] == 7

    def test_resume_flag_passed_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._config(tmp_path)
        data = tmp_path / "train.jsonl"
        data.write_text('{"text": "hi"}\n')
        mock_model = MagicMock()

        with patch("pyrecall.model.Model", return_value=mock_model):
            runner.invoke(app, ["learn", str(data), "--resume"])

        assert mock_model.learn.call_args[1]["resume"] is True

    def test_snapshot_after_triggers_snapshot_call(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._config(tmp_path)
        data = tmp_path / "train.jsonl"
        data.write_text('{"text": "hi"}\n')
        mock_model = MagicMock()
        mock_model.snapshot.return_value = _make_snapshot("post_train")

        with patch("pyrecall.model.Model", return_value=mock_model):
            runner.invoke(app, ["learn", str(data), "--snapshot-after", "post_train"])

        mock_model.snapshot.assert_called_once_with(name="post_train", tracker=None)

    def test_snapshot_after_updates_baseline_in_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._config(tmp_path)
        data = tmp_path / "train.jsonl"
        data.write_text('{"text": "hi"}\n')
        mock_model = MagicMock()
        mock_model.snapshot.return_value = _make_snapshot("post_train")

        with patch("pyrecall.model.Model", return_value=mock_model):
            runner.invoke(app, ["learn", str(data), "--snapshot-after", "post_train"])

        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["baseline_snapshot"] == "post_train"

    def test_no_snapshot_after_skips_snapshot_call(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._config(tmp_path)
        data = tmp_path / "train.jsonl"
        data.write_text('{"text": "hi"}\n')
        mock_model = MagicMock()

        with patch("pyrecall.model.Model", return_value=mock_model):
            runner.invoke(app, ["learn", str(data)])

        mock_model.snapshot.assert_not_called()

    def test_snapshot_before_triggers_snapshot_before_learn(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._config(tmp_path)
        data = tmp_path / "train.jsonl"
        data.write_text('{"text": "hi"}\n')
        mock_model = MagicMock()
        mock_model.snapshot.return_value = _make_snapshot("pre_train")

        with patch("pyrecall.model.Model", return_value=mock_model):
            runner.invoke(app, ["learn", str(data), "--snapshot-before", "pre_train"])

        # snapshot must be called before learn
        calls = mock_model.method_calls
        snapshot_idx = next(i for i, c in enumerate(calls) if c[0] == "snapshot")
        learn_idx = next(i for i, c in enumerate(calls) if c[0] == "learn")
        assert snapshot_idx < learn_idx

    def test_snapshot_before_sets_baseline_in_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._config(tmp_path)
        data = tmp_path / "train.jsonl"
        data.write_text('{"text": "hi"}\n')
        mock_model = MagicMock()
        mock_model.snapshot.return_value = _make_snapshot("pre_train")

        with patch("pyrecall.model.Model", return_value=mock_model):
            runner.invoke(app, ["learn", str(data), "--snapshot-before", "pre_train"])

        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["baseline_snapshot"] == "pre_train"

    def test_snapshot_before_and_after_both_called(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._config(tmp_path)
        data = tmp_path / "train.jsonl"
        data.write_text('{"text": "hi"}\n')
        mock_model = MagicMock()
        mock_model.snapshot.return_value = _make_snapshot("snap")

        with patch("pyrecall.model.Model", return_value=mock_model):
            runner.invoke(
                app,
                [
                    "learn",
                    str(data),
                    "--snapshot-before",
                    "pre_train",
                    "--snapshot-after",
                    "post_train",
                ],
            )

        assert mock_model.snapshot.call_count == 2
        names_called = [c.kwargs.get("name") for c in mock_model.snapshot.call_args_list]
        assert "pre_train" in names_called
        assert "post_train" in names_called

    def test_snapshot_before_and_after_baseline_is_after(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._config(tmp_path)
        data = tmp_path / "train.jsonl"
        data.write_text('{"text": "hi"}\n')
        mock_model = MagicMock()
        mock_model.snapshot.return_value = _make_snapshot("snap")

        with patch("pyrecall.model.Model", return_value=mock_model):
            runner.invoke(
                app,
                [
                    "learn",
                    str(data),
                    "--snapshot-before",
                    "pre_train",
                    "--snapshot-after",
                    "post_train",
                ],
            )

        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["baseline_snapshot"] == "post_train"

    def test_pyrecall_error_exits_with_code_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._config(tmp_path)
        data = tmp_path / "train.jsonl"
        data.write_text('{"text": "hi"}\n')
        mock_model = MagicMock()

        from pyrecall.model import PyrecallError

        mock_model.learn.side_effect = PyrecallError("bad format")

        with patch("pyrecall.model.Model", return_value=mock_model):
            result = runner.invoke(app, ["learn", str(data)])

        assert result.exit_code == 1

    def test_exit_code_zero_on_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._config(tmp_path)
        data = tmp_path / "train.jsonl"
        data.write_text('{"text": "hi"}\n')
        mock_model = MagicMock()

        with patch("pyrecall.model.Model", return_value=mock_model):
            result = runner.invoke(app, ["learn", str(data)])

        assert result.exit_code == 0

    def test_no_update_baseline_with_snapshot_after_keeps_baseline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._config(tmp_path, baseline="stable")
        data = tmp_path / "train.jsonl"
        data.write_text('{"text": "hi"}\n')
        mock_model = MagicMock()
        mock_model.snapshot.return_value = _make_snapshot("after_v1")

        with patch("pyrecall.model.Model", return_value=mock_model):
            result = runner.invoke(
                app,
                [
                    "learn",
                    str(data),
                    "--snapshot-after",
                    "after_v1",
                    "--no-update-baseline",
                ],
            )

        assert result.exit_code == 0
        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["baseline_snapshot"] == "stable"

    def test_no_update_baseline_with_snapshot_before_keeps_baseline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._config(tmp_path, baseline="stable")
        data = tmp_path / "train.jsonl"
        data.write_text('{"text": "hi"}\n')
        mock_model = MagicMock()
        mock_model.snapshot.return_value = _make_snapshot("before_v1")

        with patch("pyrecall.model.Model", return_value=mock_model):
            result = runner.invoke(
                app,
                [
                    "learn",
                    str(data),
                    "--snapshot-before",
                    "before_v1",
                    "--no-update-baseline",
                ],
            )

        assert result.exit_code == 0
        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["baseline_snapshot"] == "stable"

    def test_no_update_baseline_still_calls_learn_and_snapshot(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._config(tmp_path, baseline="stable")
        data = tmp_path / "train.jsonl"
        data.write_text('{"text": "hi"}\n')
        mock_model = MagicMock()
        mock_model.snapshot.return_value = _make_snapshot("after_v1")

        with patch("pyrecall.model.Model", return_value=mock_model):
            runner.invoke(
                app,
                [
                    "learn",
                    str(data),
                    "--snapshot-after",
                    "after_v1",
                    "--no-update-baseline",
                ],
            )

        mock_model.learn.assert_called_once()
        mock_model.snapshot.assert_called_once_with(name="after_v1", tracker=None)


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

        with patch("pyrecall.model.Model", return_value=mock_model):
            runner.invoke(app, ["snapshot", "v1"])

        mock_model.snapshot.assert_called_once_with(name="v1", tracker=None)

    def test_updates_baseline_snapshot_in_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mock_snap = _make_snapshot("my_baseline")
        mock_model = MagicMock()
        mock_model.snapshot.return_value = mock_snap

        with patch("pyrecall.model.Model", return_value=mock_model):
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

        with patch("pyrecall.model.Model", return_value=mock_model) as mock_cls:
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

        with patch("pyrecall.model.Model", return_value=mock_model) as mock_cls:
            runner.invoke(app, ["snapshot", "v1"])

        assert mock_cls.call_args[1]["strategy"] == "lora"

    def test_no_update_baseline_flag_keeps_existing_baseline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, baseline="stable_baseline")
        mock_model = MagicMock()
        mock_model.snapshot.return_value = _make_snapshot("new_snap")

        with patch("pyrecall.model.Model", return_value=mock_model):
            result = runner.invoke(app, ["snapshot", "new_snap", "--no-update-baseline"])

        assert result.exit_code == 0
        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["baseline_snapshot"] == "stable_baseline"

    def test_no_update_baseline_still_calls_snapshot(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, baseline="stable_baseline")
        mock_model = MagicMock()
        mock_model.snapshot.return_value = _make_snapshot("new_snap")

        with patch("pyrecall.model.Model", return_value=mock_model):
            runner.invoke(app, ["snapshot", "new_snap", "--no-update-baseline"])

        mock_model.snapshot.assert_called_once_with(name="new_snap", tracker=None)

    def test_default_updates_baseline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, baseline="old")
        mock_model = MagicMock()
        mock_model.snapshot.return_value = _make_snapshot("new_snap")

        with patch("pyrecall.model.Model", return_value=mock_model):
            runner.invoke(app, ["snapshot", "new_snap"])

        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["baseline_snapshot"] == "new_snap"


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

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["check"])

        assert result.exit_code == 1

    def test_fails_with_zero_snapshots(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(snapshots=[])

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
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

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
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

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["check"])

        assert result.exit_code == 2

    def test_compares_last_two_snapshots_by_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("first", created_at=datetime(2024, 1, 1))
        snap_b = _make_snapshot("second", created_at=datetime(2024, 2, 1))
        snap_c = _make_snapshot("third", created_at=datetime(2024, 3, 1))
        # list_snapshots returns oldest-first; check should compare second and third
        mgr = _make_mock_manager(snapshots=[snap_a, snap_b, snap_c])

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
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

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
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

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
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

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["check", "--before", "snap_a", "--after", "snap_b"])

        mgr.load_snapshot.assert_any_call("snap_a")
        mgr.load_snapshot.assert_any_call("snap_b")
        assert result.exit_code == 0


# ── check --json ──────────────────────────────────────────────────────────────


class TestCheckJson:
    def test_json_flag_outputs_valid_json(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("before", {"coding": 0.8})
        snap_b = _make_snapshot("after", {"coding": 0.79})
        mgr = _make_mock_manager(snapshots=[snap_a, snap_b])

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["check", "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)

    def test_json_output_has_expected_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("before", {"coding": 0.8})
        snap_b = _make_snapshot("after", {"coding": 0.79})
        mgr = _make_mock_manager(snapshots=[snap_a, snap_b])

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["check", "--json"])

        data = json.loads(result.output)
        assert "healthy" in data
        assert "snapshot_before" in data
        assert "snapshot_after" in data
        assert "degraded_skills" in data
        assert "comparisons" in data
        assert "threshold" in data

    def test_json_healthy_true_when_no_forgetting(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("before", {"coding": 0.8})
        snap_b = _make_snapshot("after", {"coding": 0.81})
        mgr = _make_mock_manager(snapshots=[snap_a, snap_b])

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["check", "--json"])

        data = json.loads(result.output)
        assert data["healthy"] is True
        assert data["degraded_skills"] == []

    def test_json_healthy_false_when_forgetting(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("before", {"coding": 0.9})
        snap_b = _make_snapshot("after", {"coding": 0.5})
        mgr = _make_mock_manager(snapshots=[snap_a, snap_b])

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["check", "--json"])

        assert result.exit_code == 2
        data = json.loads(result.output)
        assert data["healthy"] is False
        assert "coding" in data["degraded_skills"]

    def test_json_comparisons_contain_scores(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("before", {"coding": 0.8})
        snap_b = _make_snapshot("after", {"coding": 0.75})
        mgr = _make_mock_manager(snapshots=[snap_a, snap_b])

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["check", "--json"])

        data = json.loads(result.output)
        comp = data["comparisons"][0]
        assert "score_before" in comp
        assert "score_after" in comp
        assert "delta" in comp
        assert "pct_change" in comp
        assert "status" in comp

    def test_json_exit_code_still_two_on_forgetting(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("before", {"safety": 0.9})
        snap_b = _make_snapshot("after", {"safety": 0.5})
        mgr = _make_mock_manager(snapshots=[snap_a, snap_b])

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["check", "--json"])

        assert result.exit_code == 2


# ── diff ──────────────────────────────────────────────────────────────────────


class TestDiff:
    def test_fails_without_config_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["diff", "snap1", "snap2"])
        assert result.exit_code == 1

    def test_exit_code_zero_when_no_forgetting(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("snap_a", {"reasoning": 0.80})
        snap_b = _make_snapshot("snap_b", {"reasoning": 0.82})
        mgr = _make_mock_manager(
            snapshots=[snap_a, snap_b],
            snapshot_map={"snap_a": snap_a, "snap_b": snap_b},
        )

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["diff", "snap_a", "snap_b"])

        assert result.exit_code == 0

    def test_exit_code_two_when_forgetting_detected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("snap_a", {"coding": 0.90})
        snap_b = _make_snapshot("snap_b", {"coding": 0.50})
        mgr = _make_mock_manager(
            snapshots=[snap_a, snap_b],
            snapshot_map={"snap_a": snap_a, "snap_b": snap_b},
        )

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["diff", "snap_a", "snap_b"])

        assert result.exit_code == 2

    def test_fails_when_snap1_not_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager()
        mgr.load_snapshot.side_effect = FileNotFoundError("missing")

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["diff", "missing", "snap_b"])

        assert result.exit_code == 1
        assert "missing" in result.output

    def test_fails_when_snap2_not_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("snap_a", {"coding": 0.8})
        mgr = _make_mock_manager(snapshots=[snap_a], snapshot_map={"snap_a": snap_a})

        def _load(name: str) -> object:
            if name == "snap_a":
                return snap_a
            raise FileNotFoundError(name)

        mgr.load_snapshot.side_effect = _load

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["diff", "snap_a", "missing"])

        assert result.exit_code == 1
        assert "missing" in result.output

    def test_loads_both_named_snapshots(
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

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            runner.invoke(app, ["diff", "snap_a", "snap_b"])

        mgr.load_snapshot.assert_any_call("snap_a")
        mgr.load_snapshot.assert_any_call("snap_b")

    def test_json_flag_outputs_valid_json(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("snap_a", {"coding": 0.8})
        snap_b = _make_snapshot("snap_b", {"coding": 0.79})
        mgr = _make_mock_manager(
            snapshots=[snap_a, snap_b],
            snapshot_map={"snap_a": snap_a, "snap_b": snap_b},
        )

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["diff", "snap_a", "snap_b", "--json"])

        parsed = json.loads(result.output)
        assert "comparisons" in parsed

    def test_custom_threshold_respected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        # 5% drop — below default threshold (10%) but above a 3% custom threshold
        snap_a = _make_snapshot("snap_a", {"reasoning": 0.80})
        snap_b = _make_snapshot("snap_b", {"reasoning": 0.74})
        mgr = _make_mock_manager(
            snapshots=[snap_a, snap_b],
            snapshot_map={"snap_a": snap_a, "snap_b": snap_b},
        )

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["diff", "snap_a", "snap_b", "--threshold", "0.03"])

        assert result.exit_code == 2

    def test_invalid_threshold_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("snap_a")
        snap_b = _make_snapshot("snap_b")
        mgr = _make_mock_manager(
            snapshots=[snap_a, snap_b],
            snapshot_map={"snap_a": snap_a, "snap_b": snap_b},
        )

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["diff", "snap_a", "snap_b", "--threshold", "1.5"])

        assert result.exit_code == 1

    def test_does_not_require_model_to_be_loaded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("snap_a", {"coding": 0.8})
        snap_b = _make_snapshot("snap_b", {"coding": 0.79})
        mgr = _make_mock_manager(
            snapshots=[snap_a, snap_b],
            snapshot_map={"snap_a": snap_a, "snap_b": snap_b},
        )

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            with patch("pyrecall.model.Model") as mock_model_cls:
                runner.invoke(app, ["diff", "snap_a", "snap_b"])

        mock_model_cls.assert_not_called()


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

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["rollback", "nonexistent"])

        assert result.exit_code == 1

    def test_error_message_lists_available_when_not_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap = _make_snapshot("real_snap")
        mgr = _make_mock_manager(snapshots=[snap], snapshot_map={"real_snap": snap})

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["rollback", "ghost"])

        assert "ghost" in result.output

    def test_success_updates_baseline_in_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, baseline="old_snap")
        snap = _make_snapshot("new_snap")
        mgr = _make_mock_manager(snapshots=[snap], snapshot_map={"new_snap": snap})

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["rollback", "new_snap"])

        assert result.exit_code == 0
        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["baseline_snapshot"] == "new_snap"

    def test_success_exit_code_zero(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap = _make_snapshot("target")
        mgr = _make_mock_manager(snapshots=[snap], snapshot_map={"target": snap})

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["rollback", "target"])

        assert result.exit_code == 0

    def test_success_output_contains_snapshot_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap = _make_snapshot("v2")
        mgr = _make_mock_manager(snapshots=[snap], snapshot_map={"v2": snap})

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["rollback", "v2"])

        assert "v2" in result.output

    def test_old_baseline_replaced(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, baseline="old")
        snap = _make_snapshot("new")
        mgr = _make_mock_manager(snapshots=[snap], snapshot_map={"new": snap})

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            runner.invoke(app, ["rollback", "new"])

        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["baseline_snapshot"] == "new"
        assert config["baseline_snapshot"] != "old"


# ── delete ────────────────────────────────────────────────────────────────────


class TestDelete:
    def test_fails_without_config_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["delete", "v1", "--yes"])
        assert result.exit_code == 1

    def test_fails_when_snapshot_not_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(snapshots=[])

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["delete", "ghost", "--yes"])

        assert result.exit_code == 1

    def test_error_output_contains_snapshot_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(snapshots=[])

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["delete", "ghost", "--yes"])

        assert "ghost" in result.output

    def test_delete_calls_manager_delete(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap = _make_snapshot("v1")
        mgr = _make_mock_manager(snapshots=[snap], snapshot_map={"v1": snap})

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            runner.invoke(app, ["delete", "v1", "--yes"])

        mgr.delete_snapshot.assert_called_once_with("v1")

    def test_success_exit_code_zero(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap = _make_snapshot("v1")
        mgr = _make_mock_manager(snapshots=[snap], snapshot_map={"v1": snap})

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["delete", "v1", "--yes"])

        assert result.exit_code == 0


# ── replay ─────────────────────────────────────────────────────────────────────


def _write_config_with_replay(tmp_path: Path, replay_buffer_size: int = 500) -> None:
    config = {
        "model_name": "test/model",
        "strategy": "lora",
        "created_at": datetime.now().isoformat(),
        "baseline_snapshot": None,
        "replay_buffer_size": replay_buffer_size,
    }
    (tmp_path / _CONFIG_FILE).write_text(json.dumps(config, indent=2))


class TestReplayStatus:
    def test_fails_without_config_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["replay", "status"])
        assert result.exit_code == 1

    def test_disabled_message_when_buffer_size_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config_with_replay(tmp_path, replay_buffer_size=0)
        result = runner.invoke(app, ["replay", "status"])
        assert result.exit_code == 0
        assert "disabled" in result.output.lower()

    def test_shows_model_name(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config_with_replay(tmp_path)
        from unittest.mock import MagicMock

        mock_buf = MagicMock()
        mock_buf.__len__ = lambda self: 42
        mock_buf.total_seen = 100
        mock_buf.max_size = 500
        with patch("pyrecall.replay.ReplayBuffer", return_value=mock_buf):
            result = runner.invoke(app, ["replay", "status"])
        assert "test/model" in result.output

    def test_shows_fill_level(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config_with_replay(tmp_path)
        mock_buf = MagicMock()
        mock_buf.__len__ = lambda self: 250
        mock_buf.total_seen = 300
        mock_buf.max_size = 500
        with patch("pyrecall.replay.ReplayBuffer", return_value=mock_buf):
            result = runner.invoke(app, ["replay", "status"])
        assert "250" in result.output
        assert "500" in result.output

    def test_empty_buffer_note(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config_with_replay(tmp_path)
        mock_buf = MagicMock()
        mock_buf.__len__ = lambda self: 0
        mock_buf.total_seen = 0
        mock_buf.max_size = 500
        with patch("pyrecall.replay.ReplayBuffer", return_value=mock_buf):
            result = runner.invoke(app, ["replay", "status"])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_exit_code_zero_on_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config_with_replay(tmp_path)
        mock_buf = MagicMock()
        mock_buf.__len__ = lambda self: 10
        mock_buf.total_seen = 10
        mock_buf.max_size = 500
        with patch("pyrecall.replay.ReplayBuffer", return_value=mock_buf):
            result = runner.invoke(app, ["replay", "status"])
        assert result.exit_code == 0


class TestReplayClear:
    def test_fails_without_config_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["replay", "clear", "--yes"])
        assert result.exit_code == 1

    def test_already_empty_skips_prompt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config_with_replay(tmp_path)
        mock_buf = MagicMock()
        mock_buf.__len__ = lambda self: 0
        with patch("pyrecall.replay.ReplayBuffer", return_value=mock_buf):
            result = runner.invoke(app, ["replay", "clear", "--yes"])
        mock_buf.clear.assert_not_called()
        assert result.exit_code == 0

    def test_clear_called_with_yes_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config_with_replay(tmp_path)
        mock_buf = MagicMock()
        mock_buf.__len__ = lambda self: 50
        with patch("pyrecall.replay.ReplayBuffer", return_value=mock_buf):
            result = runner.invoke(app, ["replay", "clear", "--yes"])
        mock_buf.clear.assert_called_once()
        assert result.exit_code == 0

    def test_aborted_without_yes_does_not_clear(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config_with_replay(tmp_path)
        mock_buf = MagicMock()
        mock_buf.__len__ = lambda self: 50
        with patch("pyrecall.replay.ReplayBuffer", return_value=mock_buf):
            result = runner.invoke(app, ["replay", "clear"], input="n\n")
        mock_buf.clear.assert_not_called()
        assert result.exit_code == 0

    def test_success_output_contains_model_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config_with_replay(tmp_path)
        mock_buf = MagicMock()
        mock_buf.__len__ = lambda self: 20
        with patch("pyrecall.replay.ReplayBuffer", return_value=mock_buf):
            result = runner.invoke(app, ["replay", "clear", "--yes"])
        assert "test/model" in result.output

    def test_short_flag_y_skips_prompt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config_with_replay(tmp_path)
        mock_buf = MagicMock()
        mock_buf.__len__ = lambda self: 10
        with patch("pyrecall.replay.ReplayBuffer", return_value=mock_buf):
            result = runner.invoke(app, ["replay", "clear", "-y"])
        mock_buf.clear.assert_called_once()
        assert result.exit_code == 0

    def test_success_output_contains_snapshot_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap = _make_snapshot("v2")
        mgr = _make_mock_manager(snapshots=[snap], snapshot_map={"v2": snap})

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["delete", "v2", "--yes"])

        assert "v2" in result.output

    def test_deleting_baseline_clears_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, baseline="current_base")
        snap = _make_snapshot("current_base")
        mgr = _make_mock_manager(snapshots=[snap], snapshot_map={"current_base": snap})

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["delete", "current_base", "--yes"])

        assert result.exit_code == 0
        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["baseline_snapshot"] is None

    def test_deleting_non_baseline_preserves_baseline_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, baseline="keeper")
        snap = _make_snapshot("other")
        mgr = _make_mock_manager(snapshots=[snap], snapshot_map={"other": snap})

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            runner.invoke(app, ["delete", "other", "--yes"])

        config = json.loads((tmp_path / _CONFIG_FILE).read_text())
        assert config["baseline_snapshot"] == "keeper"

    def test_aborted_without_yes_does_not_delete(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap = _make_snapshot("v1")
        mgr = _make_mock_manager(snapshots=[snap], snapshot_map={"v1": snap})

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            # Simulate user typing "n" at the confirmation prompt
            result = runner.invoke(app, ["delete", "v1"], input="n\n")

        mgr.delete_snapshot.assert_not_called()
        assert result.exit_code == 0

    def test_delete_short_flag_y_skips_prompt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap = _make_snapshot("v1")
        mgr = _make_mock_manager(snapshots=[snap], snapshot_map={"v1": snap})

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["delete", "v1", "-y"])

        mgr.delete_snapshot.assert_called_once_with("v1")
        assert result.exit_code == 0


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

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "No snapshots" in result.output

    def test_exit_code_zero_with_snapshots(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, model="test/model")
        mgr = _make_mock_manager(snapshots=[_make_snapshot("v1")])

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0

    def test_snapshot_name_appears_in_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(snapshots=[_make_snapshot("release_v3")])

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["status"])

        assert "release_v3" in result.output

    def test_multiple_snapshot_names_in_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snaps = [_make_snapshot("alpha"), _make_snapshot("beta"), _make_snapshot("gamma")]
        mgr = _make_mock_manager(snapshots=snaps)

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
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

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["status"])

        # The baseline marker (★) should appear somewhere in output
        assert "★" in result.output or "v1" in result.output

    def test_list_snapshots_called_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(snapshots=[])

        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            runner.invoke(app, ["status"])

        mgr.list_snapshots.assert_called_once()


# ── history ───────────────────────────────────────────────────────────────────


class TestHistory:
    def test_fails_without_config_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["history"])
        assert result.exit_code == 1

    def test_no_snapshots_message(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(snapshots=[])
        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["history"])
        assert result.exit_code == 0
        assert "No snapshots" in result.output

    def test_single_snapshot_prompts_for_more(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(snapshots=[_make_snapshot("only")])
        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["history"])
        assert result.exit_code == 0
        assert "at least two" in result.output.lower() or "one snapshot" in result.output.lower()

    def test_shows_snapshot_names(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snaps = [_make_snapshot("before"), _make_snapshot("after")]
        mgr = _make_mock_manager(snapshots=snaps)
        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["history"])
        assert "before" in result.output
        assert "after" in result.output

    def test_shows_trend_arrows(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("v1", {"coding": 0.90})
        snap_b = _make_snapshot("v2", {"coding": 0.70})
        mgr = _make_mock_manager(snapshots=[snap_a, snap_b])
        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["history"])
        assert result.exit_code == 0
        assert "↓" in result.output

    def test_improvement_shows_up_arrow(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("v1", {"coding": 0.60})
        snap_b = _make_snapshot("v2", {"coding": 0.90})
        mgr = _make_mock_manager(snapshots=[snap_a, snap_b])
        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["history"])
        assert "↑" in result.output

    def test_category_filter(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("v1", {"coding": 0.8, "reasoning": 0.7})
        snap_b = _make_snapshot("v2", {"coding": 0.9, "reasoning": 0.6})
        mgr = _make_mock_manager(snapshots=[snap_a, snap_b])
        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["history", "--category", "coding"])
        assert result.exit_code == 0
        assert "coding" in result.output.lower() or "Coding" in result.output

    def test_invalid_category_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("v1", {"coding": 0.8})
        snap_b = _make_snapshot("v2", {"coding": 0.9})
        mgr = _make_mock_manager(snapshots=[snap_a, snap_b])
        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["history", "--category", "nonexistent"])
        assert result.exit_code == 1

    def test_last_flag_limits_snapshots(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snaps = [_make_snapshot(f"v{i}", {"coding": 0.5 + i * 0.05}) for i in range(5)]
        mgr = _make_mock_manager(snapshots=snaps)
        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["history", "--last", "2"])
        assert result.exit_code == 0
        assert "v3" in result.output
        assert "v4" in result.output
        assert "v0" not in result.output

    def test_summary_line_shown(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snap_a = _make_snapshot("first", {"coding": 0.8})
        snap_b = _make_snapshot("last", {"coding": 0.7})
        mgr = _make_mock_manager(snapshots=[snap_a, snap_b])
        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["history"])
        assert "first" in result.output
        assert "last" in result.output


# ── export ────────────────────────────────────────────────────────────────────


class TestExport:
    def test_fails_without_config_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["export"])
        assert result.exit_code == 1

    def test_no_snapshots_exits_zero_with_message(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(snapshots=[])
        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["export"])
        assert result.exit_code == 0
        assert "No snapshots" in result.output

    def test_stdout_json_is_valid(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snaps = [_make_snapshot("v1", {"coding": 0.8}), _make_snapshot("v2", {"coding": 0.75})]
        mgr = _make_mock_manager(snapshots=snaps)
        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["export"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_json_record_has_expected_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(snapshots=[_make_snapshot("v1", {"coding": 0.8})])
        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["export"])
        record = json.loads(result.output)[0]
        assert "name" in record
        assert "created_at" in record
        assert "overall" in record
        assert "categories" in record

    def test_export_json_to_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(snapshots=[_make_snapshot("v1", {"coding": 0.8})])
        out = tmp_path / "out.json"
        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["export", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        parsed = json.loads(out.read_text())
        assert parsed[0]["name"] == "v1"

    def test_export_csv_to_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        snaps = [_make_snapshot("v1", {"coding": 0.8}), _make_snapshot("v2", {"coding": 0.75})]
        mgr = _make_mock_manager(snapshots=snaps)
        out = tmp_path / "out.csv"
        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["export", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        lines = out.read_text().splitlines()
        assert lines[0].startswith("snapshot,created_at,overall")
        assert len(lines) == 3  # header + 2 rows

    def test_export_csv_stdout_via_format_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(snapshots=[_make_snapshot("v1", {"coding": 0.8})])
        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["export", "--format", "csv"])
        assert result.exit_code == 0
        assert "snapshot" in result.output
        assert "v1" in result.output

    def test_unknown_extension_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(snapshots=[_make_snapshot("v1")])
        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["export", str(tmp_path / "out.txt")])
        assert result.exit_code == 1

    def test_invalid_format_flag_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(snapshots=[_make_snapshot("v1")])
        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["export", "--format", "parquet"])
        assert result.exit_code == 1

    def test_json_categories_contain_scores(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)
        mgr = _make_mock_manager(
            snapshots=[_make_snapshot("v1", {"coding": 0.82, "reasoning": 0.77})]
        )
        with patch("pyrecall.rollback.RollbackManager", return_value=mgr):
            result = runner.invoke(app, ["export"])
        record = json.loads(result.output)[0]
        assert "coding" in record["categories"]
        assert "reasoning" in record["categories"]


# ── live ───────────────────────────────────────────────────────────────────────


def _seed_live_db(db_path: Path, pending: int = 3, trained: int = 2) -> None:
    """Create a live_data.db with a known number of pending and trained rows."""
    import sqlite3

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS interactions "
        "(id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT, response TEXT, "
        "timestamp TEXT, trained INTEGER NOT NULL DEFAULT 0)"
    )
    for i in range(pending):
        conn.execute(
            "INSERT INTO interactions (prompt, response, timestamp, trained) VALUES (?,?,?,0)",
            (f"prompt_{i}", f"response_{i}", "2024-01-01T00:00:00"),
        )
    for i in range(trained):
        conn.execute(
            "INSERT INTO interactions (prompt, response, timestamp, trained) VALUES (?,?,?,1)",
            (f"trained_prompt_{i}", f"trained_response_{i}", "2024-01-02T00:00:00"),
        )
    conn.commit()
    conn.close()


class TestLive:
    def test_status_no_db(self, tmp_path: Path) -> None:
        fake_home = tmp_path
        with patch("pyrecall.cli._default_live_db", return_value=fake_home / "missing.db"):
            result = runner.invoke(app, ["live", "status"])
        assert result.exit_code == 0
        assert "No live-learning database found" in result.output

    def test_status_shows_counts(self, tmp_path: Path) -> None:
        db = tmp_path / ".pyrecall" / "live_data.db"
        _seed_live_db(db, pending=3, trained=2)
        with patch("pyrecall.cli._default_live_db", return_value=db):
            result = runner.invoke(app, ["live", "status"])
        assert result.exit_code == 0
        assert "5" in result.output  # total
        assert "3" in result.output  # pending
        assert "2" in result.output  # trained

    def test_clear_pending_only(self, tmp_path: Path) -> None:
        db = tmp_path / ".pyrecall" / "live_data.db"
        _seed_live_db(db, pending=3, trained=2)
        with patch("pyrecall.cli._default_live_db", return_value=db):
            result = runner.invoke(app, ["live", "clear", "--yes"])
        assert result.exit_code == 0
        assert "3" in result.output

        import sqlite3

        conn = sqlite3.connect(db)
        remaining = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        trained_remaining = conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE trained=1"
        ).fetchone()[0]
        conn.close()
        assert remaining == 2
        assert trained_remaining == 2

    def test_clear_all(self, tmp_path: Path) -> None:
        db = tmp_path / ".pyrecall" / "live_data.db"
        _seed_live_db(db, pending=2, trained=3)
        with patch("pyrecall.cli._default_live_db", return_value=db):
            result = runner.invoke(app, ["live", "clear", "--all", "--yes"])
        assert result.exit_code == 0

        import sqlite3

        conn = sqlite3.connect(db)
        remaining = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        conn.close()
        assert remaining == 0

    def test_clear_no_db(self, tmp_path: Path) -> None:
        with patch("pyrecall.cli._default_live_db", return_value=tmp_path / "missing.db"):
            result = runner.invoke(app, ["live", "clear", "--yes"])
        assert result.exit_code == 0
        assert "nothing to clear" in result.output

    def test_clear_aborted_without_yes(self, tmp_path: Path) -> None:
        db = tmp_path / ".pyrecall" / "live_data.db"
        _seed_live_db(db, pending=1, trained=0)
        with patch("pyrecall.cli._default_live_db", return_value=db):
            result = runner.invoke(app, ["live", "clear"], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output

        import sqlite3

        conn = sqlite3.connect(db)
        remaining = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        conn.close()
        assert remaining == 1

    def test_clear_empty_pending_noop(self, tmp_path: Path) -> None:
        db = tmp_path / ".pyrecall" / "live_data.db"
        _seed_live_db(db, pending=0, trained=2)
        with patch("pyrecall.cli._default_live_db", return_value=db):
            result = runner.invoke(app, ["live", "clear", "--yes"])
        assert result.exit_code == 0
        assert "No" in result.output
