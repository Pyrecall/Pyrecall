"""Hugging Face Hub push/pull helpers for Pyrecall snapshots."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from .snapshot import SkillSnapshot
from .utils import get_logger, safe_model_name

logger = get_logger(__name__)

_HUB_SNAPSHOT_FILE = "snapshot.json"
_HUB_ADAPTER_DIR = "adapter"
_HUB_META_FILE = "pyrecall_meta.json"


def _require_hub():
    """Return huggingface_hub module or raise a helpful ImportError."""
    try:
        import huggingface_hub

        return huggingface_hub
    except ImportError as exc:
        raise ImportError(
            "Pushing/pulling snapshots to Hugging Face Hub requires the "
            "'huggingface_hub' package.\n"
            "Install it with: pip install huggingface_hub\n"
            "Then authenticate with: huggingface-cli login"
        ) from exc


def push_snapshot(
    snap_dir: Path,
    snapshot: SkillSnapshot,
    repo_id: str,
    *,
    include_weights: bool = True,
    private: bool = False,
) -> str:
    """Upload a snapshot to a HuggingFace Hub dataset repo.

    Creates the repo if it doesn't exist. Returns the URL of the uploaded
    snapshot folder on the Hub.

    Args:
        snap_dir: Local snapshot directory (contains snapshot.json and adapter/).
        snapshot: The SkillSnapshot object for this snapshot.
        repo_id: Hub repo in ``"owner/repo-name"`` format.
        include_weights: When False, only scores are uploaded (no adapter weights).
        private: Create the Hub repo as private if it doesn't exist yet.

    Returns:
        URL string pointing to the snapshot folder on the Hub.
    """
    hf = _require_hub()
    api = hf.HfApi()

    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)

    # Namespace on Hub: <safe_model_name>/<snapshot_name>/
    hub_prefix = f"{safe_model_name(snapshot.model_name)}/{snapshot.name}"

    with tempfile.TemporaryDirectory(prefix="pyrecall_hub_push_") as tmp:
        tmp_path = Path(tmp)

        # Always upload snapshot.json
        src_json = snap_dir / _HUB_SNAPSHOT_FILE
        if not src_json.exists():
            raise FileNotFoundError(f"snapshot.json not found in '{snap_dir}'")
        (tmp_path / _HUB_SNAPSHOT_FILE).write_bytes(src_json.read_bytes())

        # Write a pyrecall_meta.json with source info
        meta = {
            "pyrecall_hub": True,
            "snapshot_name": snapshot.name,
            "model_name": snapshot.model_name,
            "has_weights": include_weights,
            "adapter_compression": snapshot.adapter_compression,
        }
        (tmp_path / _HUB_META_FILE).write_text(json.dumps(meta, indent=2))

        # Optionally upload adapter weights
        if include_weights:
            adapter_dir = snap_dir / _HUB_ADAPTER_DIR
            if adapter_dir.exists():
                adapter_tmp = tmp_path / _HUB_ADAPTER_DIR
                shutil.copytree(adapter_dir, adapter_tmp)
            else:
                logger.warning(
                    "include_weights=True but no adapter/ directory found in '%s'. "
                    "Uploading scores only.",
                    snap_dir,
                )

        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=hub_prefix,
        )

    url = f"{api.endpoint}/datasets/{repo_id}/tree/main/{hub_prefix}"
    logger.debug("Snapshot '%s' pushed to %s", snapshot.name, url)
    return url


def pull_snapshot(
    name: str,
    model_name: str,
    repo_id: str,
    dest_dir: Path,
    *,
    include_weights: bool = True,
) -> SkillSnapshot:
    """Download a snapshot from a HuggingFace Hub dataset repo.

    Downloads into *dest_dir*/<name>/ (the standard RollbackManager layout).
    Returns the loaded SkillSnapshot.

    Args:
        name: Snapshot name on the Hub (used as the folder name).
        model_name: Model name — used to build the Hub namespace path.
        repo_id: Hub repo in ``"owner/repo-name"`` format.
        dest_dir: Local directory where the snapshot folder will be created
                  (should be the RollbackManager base_dir for this model).
        include_weights: When False, skip downloading the adapter/ directory.

    Returns:
        The loaded :class:`~pyrecall.snapshot.SkillSnapshot`.
    """
    hf = _require_hub()

    hub_prefix = f"{safe_model_name(model_name)}/{name}"
    snap_dest = dest_dir / name
    snap_dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pyrecall_hub_pull_") as tmp:
        tmp_path = Path(tmp)

        # Always download snapshot.json
        try:
            hf.hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=f"{hub_prefix}/{_HUB_SNAPSHOT_FILE}",
                local_dir=str(tmp_path),
            )
        except Exception as exc:
            raise FileNotFoundError(
                f"Snapshot '{name}' not found in Hub repo '{repo_id}'. "
                f"Hub path: {hub_prefix}/{_HUB_SNAPSHOT_FILE}\n"
                f"Original error: {exc}"
            ) from exc

        # Read meta to check if weights exist on Hub (optional — older pushes may not have it)
        meta: dict = {}
        try:
            hf.hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=f"{hub_prefix}/{_HUB_META_FILE}",
                local_dir=str(tmp_path),
            )
            meta_path = tmp_path / hub_prefix / _HUB_META_FILE
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
        except Exception as _meta_exc:
            logger.debug(
                "Could not fetch pyrecall_meta.json for snapshot '%s' (treating as weights-included): %s",
                name,
                _meta_exc,
            )

        hub_has_weights = meta.get("has_weights", True)

        # Copy snapshot.json to dest
        downloaded_json = tmp_path / hub_prefix / _HUB_SNAPSHOT_FILE
        (snap_dest / _HUB_SNAPSHOT_FILE).write_bytes(downloaded_json.read_bytes())

        # Download adapter weights if requested and available
        if include_weights and hub_has_weights:
            try:
                hf.snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    allow_patterns=[f"{hub_prefix}/{_HUB_ADAPTER_DIR}/**"],
                    local_dir=str(tmp_path),
                )
                adapter_tmp = tmp_path / hub_prefix / _HUB_ADAPTER_DIR
                adapter_dest = snap_dest / _HUB_ADAPTER_DIR
                if adapter_dest.exists():
                    shutil.rmtree(adapter_dest)
                if adapter_tmp.exists():
                    shutil.copytree(adapter_tmp, adapter_dest)
                    logger.debug("Adapter weights downloaded to %s", adapter_dest)
                else:
                    logger.warning(
                        "Adapter directory not found on Hub for snapshot '%s'; "
                        "scores downloaded but weights unavailable.",
                        name,
                    )
            except Exception as exc:
                logger.warning(
                    "Could not download adapter weights for snapshot '%s': %s. "
                    "Scores were downloaded successfully.",
                    name,
                    exc,
                )
        elif include_weights and not hub_has_weights:
            logger.info(
                "Snapshot '%s' was pushed without weights (include_weights=False). Scores only.",
                name,
            )

    snap = SkillSnapshot.load(snap_dest)
    # Fixup adapter_path to point at the local copy (the stored path is from the
    # original machine and won't exist locally).
    local_adapter = snap_dest / _HUB_ADAPTER_DIR
    snap.adapter_path = local_adapter if local_adapter.exists() else None
    snap.hub_repo = repo_id
    # Persist hub_repo into the local snapshot.json so status shows [hub].
    snap.save(snap_dest)
    logger.debug("Snapshot '%s' pulled from Hub repo '%s'", name, repo_id)
    return snap
