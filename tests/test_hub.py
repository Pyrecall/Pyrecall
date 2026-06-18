"""Tests for pyrecall.hub push/pull helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyrecall.hub import _HUB_META_FILE, pull_snapshot, push_snapshot
from pyrecall.snapshot import SkillScore, SkillSnapshot


def _make_snapshot(name: str = "before_v1", model: str = "test/model") -> SkillSnapshot:
    return SkillSnapshot(
        name=name,
        model_name=model,
        scores=[SkillScore(category="math", prompt="1+1", response="2", score=0.9)],
    )


def _write_snap_dir(tmp_path: Path, snap: SkillSnapshot) -> Path:
    """Write a minimal snapshot directory as RollbackManager would."""
    snap_dir = tmp_path / snap.name
    snap_dir.mkdir()
    snap.save(snap_dir)
    return snap_dir


class TestRequireHub:
    def test_raises_import_error_when_missing(self):
        from pyrecall.hub import _require_hub

        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="huggingface_hub"):
                _require_hub()


class TestPushSnapshot:
    def test_raises_if_snapshot_json_missing(self, tmp_path):
        snap = _make_snapshot()
        empty_dir = tmp_path / "before_v1"
        empty_dir.mkdir()

        mock_hf = MagicMock()
        with patch("pyrecall.hub._require_hub", return_value=mock_hf):
            with pytest.raises(FileNotFoundError, match="snapshot.json"):
                push_snapshot(empty_dir, snap, "owner/repo")

    def _capture_upload(self) -> tuple[MagicMock, list[dict]]:
        """Return (mock_hf, captured_calls) where captured_calls[i] has 'folder_path'
        and 'files' (list of relative paths present at call time)."""
        captured: list[dict] = []

        def fake_upload_folder(**kwargs):
            folder = Path(kwargs["folder_path"])
            captured.append(
                {
                    "folder_path": folder,
                    "files": [str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()],
                    "kwargs": kwargs,
                }
            )

        mock_api = MagicMock()
        mock_api.upload_folder.side_effect = fake_upload_folder
        mock_hf = MagicMock()
        mock_hf.HfApi.return_value = mock_api
        return mock_hf, captured

    def test_uploads_snapshot_json_and_meta(self, tmp_path):
        snap = _make_snapshot()
        snap_dir = _write_snap_dir(tmp_path, snap)

        mock_hf, captured = self._capture_upload()
        with patch("pyrecall.hub._require_hub", return_value=mock_hf):
            url = push_snapshot(snap_dir, snap, "owner/repo")

        assert mock_hf.HfApi.return_value.create_repo.called
        assert len(captured) == 1
        files = captured[0]["files"]
        assert "snapshot.json" in files
        assert "pyrecall_meta.json" in files
        assert "owner/repo" in url and "before_v1" in url

    def test_no_weights_skips_adapter(self, tmp_path):
        snap = _make_snapshot()
        snap_dir = _write_snap_dir(tmp_path, snap)
        (snap_dir / "adapter").mkdir()
        (snap_dir / "adapter" / "config.json").write_text("{}")

        mock_hf, captured = self._capture_upload()
        with patch("pyrecall.hub._require_hub", return_value=mock_hf):
            push_snapshot(snap_dir, snap, "owner/repo", include_weights=False)

        files = captured[0]["files"]
        assert not any("adapter" in f for f in files)

    def test_includes_adapter_weights_when_present(self, tmp_path):
        snap = _make_snapshot()
        snap_dir = _write_snap_dir(tmp_path, snap)
        adapter_dir = snap_dir / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.bin").write_bytes(b"fake-weights")

        mock_hf, captured = self._capture_upload()
        with patch("pyrecall.hub._require_hub", return_value=mock_hf):
            push_snapshot(snap_dir, snap, "owner/repo", include_weights=True)

        files = captured[0]["files"]
        assert any("adapter_model.bin" in f for f in files)


class TestPullSnapshot:
    def _setup_mock_hf(self, tmp_path: Path, snap: SkillSnapshot, hub_prefix: str):
        """Return a mock hf module that serves snap's JSON from tmp_path."""
        snap_src = tmp_path / "hub_source" / hub_prefix
        snap_src.mkdir(parents=True)
        snap.save(snap_src)

        def fake_hf_hub_download(*, repo_id, repo_type, filename, local_dir, **kw):
            # filename is like "<hub_prefix>/snapshot.json"
            rel = Path(filename)
            src = tmp_path / "hub_source" / rel
            dst = Path(local_dir) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())

        mock_hf = MagicMock()
        mock_hf.hf_hub_download.side_effect = fake_hf_hub_download
        # snapshot_download does nothing (no adapter in this helper)
        mock_hf.snapshot_download.return_value = str(tmp_path / "hub_source")
        return mock_hf

    def test_downloads_snapshot_json(self, tmp_path):
        snap = _make_snapshot()
        from pyrecall.utils import safe_model_name

        hub_prefix = f"{safe_model_name(snap.model_name)}/{snap.name}"
        mock_hf = self._setup_mock_hf(tmp_path, snap, hub_prefix)
        dest_dir = tmp_path / "local_snapshots"

        with patch("pyrecall.hub._require_hub", return_value=mock_hf):
            pulled = pull_snapshot(
                snap.name, snap.model_name, "owner/repo", dest_dir, include_weights=False
            )

        assert pulled.name == snap.name
        assert pulled.model_name == snap.model_name
        assert pulled.hub_repo == "owner/repo"

    def test_hub_repo_persisted_to_disk(self, tmp_path):
        snap = _make_snapshot()
        from pyrecall.utils import safe_model_name

        hub_prefix = f"{safe_model_name(snap.model_name)}/{snap.name}"
        mock_hf = self._setup_mock_hf(tmp_path, snap, hub_prefix)
        dest_dir = tmp_path / "local_snapshots"

        with patch("pyrecall.hub._require_hub", return_value=mock_hf):
            pull_snapshot(snap.name, snap.model_name, "owner/repo", dest_dir, include_weights=False)

        saved = json.loads((dest_dir / snap.name / "snapshot.json").read_text())
        assert saved["hub_repo"] == "owner/repo"

    def test_raises_file_not_found_on_missing_snapshot(self, tmp_path):
        def raise_exc(**kw):
            raise Exception("404 not found")

        mock_hf = MagicMock()
        mock_hf.hf_hub_download.side_effect = raise_exc
        dest_dir = tmp_path / "local"

        with patch("pyrecall.hub._require_hub", return_value=mock_hf):
            with pytest.raises(FileNotFoundError, match="before_v1"):
                pull_snapshot("before_v1", "test/model", "owner/repo", dest_dir)


class TestSnapshotHubField:
    def test_hub_repo_roundtrips_through_save_load(self, tmp_path):
        snap = _make_snapshot()
        snap.hub_repo = "owner/my-repo"
        snap.save(tmp_path)
        loaded = SkillSnapshot.load(tmp_path)
        assert loaded.hub_repo == "owner/my-repo"

    def test_hub_repo_defaults_to_none(self, tmp_path):
        snap = _make_snapshot()
        snap.save(tmp_path)
        loaded = SkillSnapshot.load(tmp_path)
        assert loaded.hub_repo is None


class TestPushPrivate:
    def test_create_repo_called_with_private_true(self, tmp_path):
        snap = _make_snapshot()
        snap_dir = _write_snap_dir(tmp_path, snap)

        mock_api = MagicMock()
        mock_hf = MagicMock()
        mock_hf.HfApi.return_value = mock_api

        with patch("pyrecall.hub._require_hub", return_value=mock_hf):
            push_snapshot(snap_dir, snap, "owner/repo", private=True)

        mock_api.create_repo.assert_called_once_with(
            repo_id="owner/repo", repo_type="dataset", exist_ok=True, private=True
        )


class TestPullOverExistingAdapter:
    def test_existing_adapter_dir_replaced(self, tmp_path):
        snap = _make_snapshot()
        from pyrecall.utils import safe_model_name

        hub_prefix = f"{safe_model_name(snap.model_name)}/{snap.name}"

        # Set up hub source with adapter weights
        snap_src = tmp_path / "hub_source" / hub_prefix
        snap_src.mkdir(parents=True)
        snap.save(snap_src)
        adapter_src = snap_src / "adapter"
        adapter_src.mkdir()
        (adapter_src / "new_weights.bin").write_bytes(b"new")

        def fake_hf_hub_download(*, repo_id, repo_type, filename, local_dir, **kw):
            rel = Path(filename)
            src = tmp_path / "hub_source" / rel
            dst = Path(local_dir) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                dst.write_bytes(src.read_bytes())

        def fake_snapshot_download(*, repo_id, repo_type, allow_patterns, local_dir, **kw):
            # Copy adapter dir into local_dir under the hub_prefix path
            adapter_dst = Path(local_dir) / hub_prefix / "adapter"
            adapter_dst.mkdir(parents=True, exist_ok=True)
            (adapter_dst / "new_weights.bin").write_bytes(b"new")

        mock_hf = MagicMock()
        mock_hf.hf_hub_download.side_effect = fake_hf_hub_download
        mock_hf.snapshot_download.side_effect = fake_snapshot_download

        dest_dir = tmp_path / "local"

        # Pre-create an existing stale adapter dir
        existing_adapter = dest_dir / snap.name / "adapter"
        existing_adapter.mkdir(parents=True)
        (existing_adapter / "old_weights.bin").write_bytes(b"old")

        with patch("pyrecall.hub._require_hub", return_value=mock_hf):
            pulled = pull_snapshot(snap.name, snap.model_name, "owner/repo", dest_dir)

        assert (dest_dir / snap.name / "adapter" / "new_weights.bin").exists()
        assert not (dest_dir / snap.name / "adapter" / "old_weights.bin").exists()
        assert pulled.adapter_path == dest_dir / snap.name / "adapter"


class TestMetaFetchLogsOnError:
    def test_auth_error_on_meta_fetch_is_logged_not_silenced(self, tmp_path, caplog):
        import logging

        snap = _make_snapshot()
        from pyrecall.utils import safe_model_name

        hub_prefix = f"{safe_model_name(snap.model_name)}/{snap.name}"
        snap_src = tmp_path / "hub_source" / hub_prefix
        snap_src.mkdir(parents=True)
        snap.save(snap_src)

        call_count = {"n": 0}

        def fake_hf_hub_download(*, repo_id, repo_type, filename, local_dir, **kw):
            call_count["n"] += 1
            if _HUB_META_FILE in filename:
                raise Exception("401 Unauthorized")
            rel = Path(filename)
            src = tmp_path / "hub_source" / rel
            dst = Path(local_dir) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())

        mock_hf = MagicMock()
        mock_hf.hf_hub_download.side_effect = fake_hf_hub_download

        dest_dir = tmp_path / "local"
        with caplog.at_level(logging.DEBUG, logger="pyrecall.hub"):
            with patch("pyrecall.hub._require_hub", return_value=mock_hf):
                pulled = pull_snapshot(
                    snap.name, snap.model_name, "owner/repo", dest_dir, include_weights=False
                )

        assert pulled.name == snap.name
        assert any("pyrecall_meta.json" in r.message or "401" in r.message for r in caplog.records)
