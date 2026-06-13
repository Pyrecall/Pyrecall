"""Tests for WandbTracker and MLflowTracker."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from pyrecall.snapshot import SkillScore, SkillSnapshot
from pyrecall.trackers import MLflowTracker, NeptuneTracker, SnapshotTracker, WandbTracker


def _make_snapshot(name: str = "test_snap") -> SkillSnapshot:
    return SkillSnapshot(
        name=name,
        model_name="test/model",
        created_at=datetime(2024, 1, 1),
        scores=[
            SkillScore(category="coding", prompt="p1", response="r1", score=0.8),
            SkillScore(category="reasoning", prompt="p2", response="r2", score=0.7),
        ],
    )


# ── protocol ──────────────────────────────────────────────────────────────────


class TestSnapshotTrackerProtocol:
    def test_wandb_tracker_satisfies_protocol(self) -> None:
        assert isinstance(WandbTracker(), SnapshotTracker)

    def test_mlflow_tracker_satisfies_protocol(self) -> None:
        assert isinstance(MLflowTracker(), SnapshotTracker)

    def test_neptune_tracker_satisfies_protocol(self) -> None:
        assert isinstance(NeptuneTracker(project="ws/proj"), SnapshotTracker)

    def test_custom_tracker_satisfies_protocol(self) -> None:
        class MyTracker:
            def log_snapshot(self, snapshot):
                pass

        assert isinstance(MyTracker(), SnapshotTracker)


# ── WandbTracker ──────────────────────────────────────────────────────────────


class TestWandbTracker:
    def test_raises_import_error_when_wandb_missing(self) -> None:
        snap = _make_snapshot()
        tracker = WandbTracker()
        with patch.dict("sys.modules", {"wandb": None}):
            with pytest.raises(ImportError, match="wandb"):
                tracker.log_snapshot(snap)

    def test_calls_wandb_init_with_snapshot_name(self) -> None:
        snap = _make_snapshot("my_snap")
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            WandbTracker(project="test-project").log_snapshot(snap)

        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["name"] == "my_snap"
        assert call_kwargs["project"] == "test-project"

    def test_logs_overall_and_category_metrics(self) -> None:
        snap = _make_snapshot()
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            WandbTracker().log_snapshot(snap)

        logged = mock_run.log.call_args[0][0]
        assert "pyrecall/overall" in logged
        assert "pyrecall/coding" in logged
        assert "pyrecall/reasoning" in logged

    def test_calls_finish(self) -> None:
        snap = _make_snapshot()
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            WandbTracker().log_snapshot(snap)

        mock_run.finish.assert_called_once()

    def test_default_project_is_pyrecall(self) -> None:
        tracker = WandbTracker()
        assert tracker.project == "pyrecall"


# ── MLflowTracker ─────────────────────────────────────────────────────────────


class TestMLflowTracker:
    def test_raises_import_error_when_mlflow_missing(self) -> None:
        snap = _make_snapshot()
        tracker = MLflowTracker()
        with patch.dict("sys.modules", {"mlflow": None}):
            with pytest.raises(ImportError, match="mlflow"):
                tracker.log_snapshot(snap)

    def test_sets_experiment_name(self) -> None:
        snap = _make_snapshot()
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            MLflowTracker(experiment_name="my-exp").log_snapshot(snap)

        mock_mlflow.set_experiment.assert_called_once_with("my-exp")

    def test_logs_metrics_with_correct_keys(self) -> None:
        snap = _make_snapshot()
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            MLflowTracker().log_snapshot(snap)

        logged = mock_mlflow.log_metrics.call_args[0][0]
        assert "pyrecall.overall" in logged
        assert "pyrecall.coding" in logged
        assert "pyrecall.reasoning" in logged

    def test_sets_snapshot_and_model_tags(self) -> None:
        snap = _make_snapshot("v1")
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            MLflowTracker().log_snapshot(snap)

        tag_calls = {c[0][0]: c[0][1] for c in mock_mlflow.set_tag.call_args_list}
        assert tag_calls["pyrecall.snapshot"] == "v1"
        assert tag_calls["pyrecall.model"] == "test/model"

    def test_sets_tracking_uri_when_provided(self) -> None:
        snap = _make_snapshot()
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            MLflowTracker(tracking_uri="http://localhost:5000").log_snapshot(snap)

        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")

    def test_does_not_set_tracking_uri_when_not_provided(self) -> None:
        snap = _make_snapshot()
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            MLflowTracker().log_snapshot(snap)

        mock_mlflow.set_tracking_uri.assert_not_called()

    def test_default_experiment_is_pyrecall(self) -> None:
        tracker = MLflowTracker()
        assert tracker.experiment_name == "pyrecall"


# ── Model.snapshot integration ────────────────────────────────────────────────


class TestModelSnapshotTrackerIntegration:
    def _make_model(self, scores):
        from pyrecall.model import Model

        m = object.__new__(Model)
        m.model_name = "test/model"
        m.model = MagicMock()
        m.rollback_manager = MagicMock()
        m._baseline_snapshot_name = None
        m._snapshot_compression = "none"
        return m

    def test_tracker_log_snapshot_called_after_save(self) -> None:
        snap = _make_snapshot("v1")
        mock_tracker = MagicMock(spec=["log_snapshot"])

        with patch("pyrecall.model.Model._run_benchmarks", return_value=snap.scores):
            m = self._make_model(snap.scores)
            result = m.snapshot("v1", tracker=mock_tracker)

        mock_tracker.log_snapshot.assert_called_once()
        assert result.name == "v1"

    def test_multiple_trackers_all_called(self) -> None:
        snap = _make_snapshot("v1")
        t1 = MagicMock(spec=["log_snapshot"])
        t2 = MagicMock(spec=["log_snapshot"])

        with patch("pyrecall.model.Model._run_benchmarks", return_value=snap.scores):
            m = self._make_model(snap.scores)
            m.snapshot("v1", tracker=[t1, t2])

        t1.log_snapshot.assert_called_once()
        t2.log_snapshot.assert_called_once()

    def test_no_tracker_does_not_error(self) -> None:
        snap = _make_snapshot("v1")

        with patch("pyrecall.model.Model._run_benchmarks", return_value=snap.scores):
            m = self._make_model(snap.scores)
            m.snapshot("v1")  # no tracker — should not raise


# ── NeptuneTracker ────────────────────────────────────────────────────────────


class TestNeptuneTracker:
    def test_raises_import_error_when_neptune_missing(self) -> None:
        snap = _make_snapshot()
        tracker = NeptuneTracker(project="ws/proj")
        with patch.dict("sys.modules", {"neptune": None}):
            with pytest.raises(ImportError, match="neptune"):
                tracker.log_snapshot(snap)

    def test_calls_init_run_with_snapshot_name_and_project(self) -> None:
        snap = _make_snapshot("my_snap")
        mock_neptune = MagicMock()
        mock_run = MagicMock()
        mock_neptune.init_run.return_value = mock_run

        with patch.dict("sys.modules", {"neptune": mock_neptune}):
            NeptuneTracker(project="ws/proj").log_snapshot(snap)

        mock_neptune.init_run.assert_called_once()
        call_kwargs = mock_neptune.init_run.call_args[1]
        assert call_kwargs["name"] == "my_snap"
        assert call_kwargs["project"] == "ws/proj"

    def test_logs_overall_and_category_metrics(self) -> None:
        snap = _make_snapshot()
        mock_neptune = MagicMock()
        mock_run = MagicMock()
        mock_neptune.init_run.return_value = mock_run

        with patch.dict("sys.modules", {"neptune": mock_neptune}):
            NeptuneTracker(project="ws/proj").log_snapshot(snap)

        assigned_keys = [call[0][0] for call in mock_run.__setitem__.call_args_list]
        assert "pyrecall/overall" in assigned_keys
        assert "pyrecall/coding" in assigned_keys
        assert "pyrecall/reasoning" in assigned_keys

    def test_includes_pyrecall_tag(self) -> None:
        snap = _make_snapshot()
        mock_neptune = MagicMock()
        mock_run = MagicMock()
        mock_neptune.init_run.return_value = mock_run

        with patch.dict("sys.modules", {"neptune": mock_neptune}):
            NeptuneTracker(project="ws/proj").log_snapshot(snap)

        call_kwargs = mock_neptune.init_run.call_args[1]
        assert "pyrecall" in call_kwargs["tags"]

    def test_calls_stop(self) -> None:
        snap = _make_snapshot()
        mock_neptune = MagicMock()
        mock_run = MagicMock()
        mock_neptune.init_run.return_value = mock_run

        with patch.dict("sys.modules", {"neptune": mock_neptune}):
            NeptuneTracker(project="ws/proj").log_snapshot(snap)

        mock_run.stop.assert_called_once()

    def test_default_project_stored(self) -> None:
        tracker = NeptuneTracker(project="workspace/myproject")
        assert tracker.project == "workspace/myproject"
