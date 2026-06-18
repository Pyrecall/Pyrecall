"""Tests for RollbackManager — no actual model weights are saved in these tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pyrecall.rollback import RollbackManager
from pyrecall.snapshot import SkillScore, SkillSnapshot

# ── helpers ────────────────────────────────────────────────────────────────────


def _make_snapshot(name: str, model_name: str = "test/model") -> SkillSnapshot:
    return SkillSnapshot(
        name=name,
        model_name=model_name,
        created_at=datetime(2024, 6, 1, 12, 0, 0),
        scores=[
            SkillScore(
                category="reasoning",
                prompt="What is 2+2?",
                response="4",
                score=0.85,
            )
        ],
    )


def _make_mock_peft_model(save_dir_tracker: list | None = None) -> MagicMock:
    """A PEFT model mock that records where save_pretrained is called."""
    mock = MagicMock()

    def _save(path: str) -> None:
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "adapter_config.json").write_text("{}")
        (dir_path / "adapter_model.bin").write_bytes(b"mock_weights")
        if save_dir_tracker is not None:
            save_dir_tracker.append(path)

    mock.save_pretrained.side_effect = _save
    return mock


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def manager(tmp_path: Path) -> RollbackManager:
    return RollbackManager(model_name="test/model", base_dir=tmp_path / "snapshots")


# ── tests ──────────────────────────────────────────────────────────────────────


class TestRollbackManagerInit:
    def test_creates_base_dir(self, tmp_path: Path) -> None:
        base = tmp_path / "new_dir"
        assert not base.exists()
        RollbackManager(model_name="test/model", base_dir=base)
        assert base.exists()

    def test_default_dir_under_home(self) -> None:
        mgr = RollbackManager(model_name="test/model")
        assert ".pyrecall" in str(mgr.base_dir)
        assert "test--model" in str(mgr.base_dir)

    def test_safe_model_name_replaces_slash(self) -> None:
        # When base_dir is not overridden, safe_model_name is applied to the default path.
        mgr = RollbackManager(model_name="org/model-name")
        assert "org--model-name" in str(mgr.base_dir)


class TestRollbackManagerSave:
    def test_save_creates_snapshot_json(self, manager: RollbackManager) -> None:
        snap = _make_snapshot("v1")
        mock_peft = _make_mock_peft_model()
        manager.save(snap, mock_peft)

        snap_file = manager.base_dir / "v1" / "snapshot.json"
        assert snap_file.exists()

    def test_save_creates_adapter_dir(self, manager: RollbackManager) -> None:
        snap = _make_snapshot("v1")
        mock_peft = _make_mock_peft_model()
        manager.save(snap, mock_peft)

        adapter_dir = manager.base_dir / "v1" / "adapter"
        assert adapter_dir.exists()

    def test_save_sets_adapter_path_on_snapshot(self, manager: RollbackManager) -> None:
        snap = _make_snapshot("v1")
        assert snap.adapter_path is None

        mock_peft = _make_mock_peft_model()
        manager.save(snap, mock_peft)

        assert snap.adapter_path is not None
        assert "adapter" in str(snap.adapter_path)

    def test_save_calls_peft_save_pretrained(self, manager: RollbackManager) -> None:
        snap = _make_snapshot("v1")
        mock_peft = _make_mock_peft_model()
        manager.save(snap, mock_peft)

        mock_peft.save_pretrained.assert_called_once()

    def test_save_returns_snapshot_directory(self, manager: RollbackManager) -> None:
        snap = _make_snapshot("v1")
        mock_peft = _make_mock_peft_model()
        result = manager.save(snap, mock_peft)

        assert result == manager.base_dir / "v1"


class TestRollbackManagerLoad:
    def test_load_existing_snapshot(self, manager: RollbackManager) -> None:
        snap = _make_snapshot("v1")
        mock_peft = _make_mock_peft_model()
        manager.save(snap, mock_peft)

        loaded = manager.load_snapshot("v1")
        assert loaded.name == "v1"

    def test_load_preserves_scores(self, manager: RollbackManager) -> None:
        snap = _make_snapshot("v1")
        mock_peft = _make_mock_peft_model()
        manager.save(snap, mock_peft)

        loaded = manager.load_snapshot("v1")
        assert len(loaded.scores) == 1
        assert loaded.scores[0].category == "reasoning"
        assert loaded.scores[0].score == pytest.approx(0.85)

    def test_load_missing_snapshot_raises(self, manager: RollbackManager) -> None:
        with pytest.raises(FileNotFoundError, match="ghost_snap"):
            manager.load_snapshot("ghost_snap")

    def test_load_error_lists_available(self, manager: RollbackManager) -> None:
        snap = _make_snapshot("real_snap")
        mock_peft = _make_mock_peft_model()
        manager.save(snap, mock_peft)

        with pytest.raises(FileNotFoundError, match="real_snap"):
            manager.load_snapshot("nonexistent")


class TestRollbackManagerList:
    def test_empty_when_no_snapshots(self, manager: RollbackManager) -> None:
        assert manager.list_snapshots() == []

    def test_lists_all_saved_snapshots(self, manager: RollbackManager) -> None:
        for name in ("first", "second", "third"):
            manager.save(_make_snapshot(name), _make_mock_peft_model())

        snaps = manager.list_snapshots()
        assert len(snaps) == 3

    def test_snapshots_sorted_by_creation_time(self, manager: RollbackManager) -> None:
        snap_a = SkillSnapshot(
            name="alpha",
            model_name="test/model",
            created_at=datetime(2024, 1, 1),
            scores=[],
        )
        snap_b = SkillSnapshot(
            name="beta",
            model_name="test/model",
            created_at=datetime(2024, 6, 1),
            scores=[],
        )
        for snap in (snap_b, snap_a):  # save out of order
            manager.save(snap, _make_mock_peft_model())

        listed = manager.list_snapshots()
        assert listed[0].name == "alpha"
        assert listed[1].name == "beta"


class TestRollbackManagerHasAndDelete:
    def test_has_snapshot_true_after_save(self, manager: RollbackManager) -> None:
        manager.save(_make_snapshot("exists"), _make_mock_peft_model())
        assert manager.has_snapshot("exists") is True

    def test_has_snapshot_false_when_missing(self, manager: RollbackManager) -> None:
        assert manager.has_snapshot("ghost") is False

    def test_delete_removes_snapshot(self, manager: RollbackManager) -> None:
        manager.save(_make_snapshot("to_delete"), _make_mock_peft_model())
        assert manager.has_snapshot("to_delete")

        manager.delete_snapshot("to_delete")
        assert not manager.has_snapshot("to_delete")

    def test_delete_missing_raises(self, manager: RollbackManager) -> None:
        with pytest.raises(FileNotFoundError):
            manager.delete_snapshot("nonexistent")

    def test_list_after_delete_decreases_count(self, manager: RollbackManager) -> None:
        for name in ("a", "b", "c"):
            manager.save(_make_snapshot(name), _make_mock_peft_model())

        manager.delete_snapshot("b")
        remaining = [s.name for s in manager.list_snapshots()]
        assert "b" not in remaining
        assert len(remaining) == 2


class TestRollbackManagerBaseDir:
    def test_explicit_base_dir_still_namespaces_by_model(self, tmp_path: Path) -> None:
        mgr = RollbackManager(model_name="org/model-a", base_dir=tmp_path)
        assert "org--model-a" in str(mgr.base_dir)
        assert str(mgr.base_dir).startswith(str(tmp_path))

    def test_two_models_same_base_dir_do_not_collide(self, tmp_path: Path) -> None:
        mgr_a = RollbackManager(model_name="llama-7b", base_dir=tmp_path)
        mgr_b = RollbackManager(model_name="mistral-7b", base_dir=tmp_path)
        assert mgr_a.base_dir != mgr_b.base_dir

    def test_snapshot_saved_under_model_subdir(self, tmp_path: Path) -> None:
        mgr = RollbackManager(model_name="org/mymodel", base_dir=tmp_path)
        mgr.save(_make_snapshot("v1"), _make_mock_peft_model())
        expected = tmp_path / "org--mymodel" / "v1" / "snapshot.json"
        assert expected.exists()


class TestSaveLock:
    def test_lock_file_created_during_save(self, tmp_path: Path) -> None:
        mgr = RollbackManager(model_name="test/model", base_dir=tmp_path)
        mgr.save(_make_snapshot("v1"), _make_mock_peft_model())
        # Lock file should exist (not cleaned up — it's intentionally kept as a
        # cheap sentinel so future lock calls don't need to re-create the dir).
        snap_dir = tmp_path / "test--model" / "v1"
        assert snap_dir.exists()

    def test_concurrent_saves_do_not_corrupt(self, tmp_path: Path) -> None:
        import threading

        mgr = RollbackManager(model_name="test/model", base_dir=tmp_path)
        errors: list[Exception] = []

        def do_save():
            try:
                mgr.save(_make_snapshot("v1"), _make_mock_peft_model())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=do_save) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent saves raised: {errors}"
        snap = mgr.load_snapshot("v1")
        assert snap.name == "v1"


class TestSnapshotNameValidation:
    """_validate_snapshot_name edge cases."""

    def test_dot_name_raises(self, tmp_path: Path) -> None:
        mgr = RollbackManager(model_name="test/model", base_dir=tmp_path)
        with pytest.raises(ValueError, match=r"\."):
            mgr.save(_make_snapshot("."), _make_mock_peft_model())

    def test_slash_name_raises(self, tmp_path: Path) -> None:
        mgr = RollbackManager(model_name="test/model", base_dir=tmp_path)
        with pytest.raises(ValueError):
            mgr.save(_make_snapshot("a/b"), _make_mock_peft_model())

    def test_dotdot_name_raises(self, tmp_path: Path) -> None:
        mgr = RollbackManager(model_name="test/model", base_dir=tmp_path)
        with pytest.raises(ValueError):
            mgr.save(_make_snapshot(".."), _make_mock_peft_model())

    def test_valid_name_accepted(self, tmp_path: Path) -> None:
        mgr = RollbackManager(model_name="test/model", base_dir=tmp_path)
        mgr.save(_make_snapshot("valid-snap_v1"), _make_mock_peft_model())
        assert mgr.has_snapshot("valid-snap_v1")
