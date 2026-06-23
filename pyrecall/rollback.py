"""RollbackManager — save and restore LoRA adapter checkpoints."""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock

from .snapshot import SkillSnapshot
from .utils import get_logger, safe_model_name

if TYPE_CHECKING:
    from peft import PeftModel

logger = get_logger(__name__)


@contextmanager
def _snap_lock(snap_dir: Path):
    """Acquire an exclusive filesystem lock for a snapshot directory.

    The lock file is placed alongside the snap directory (not inside it) so
    that shutil.rmtree on snap_dir never tries to delete a held lock file,
    which would raise PermissionError on Windows.
    """
    lock_file = snap_dir.parent / f".{snap_dir.name}.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_file)):
        yield


def _validate_snapshot_name(name: str) -> None:
    """Raise ValueError if *name* contains path separators or traversal segments."""
    if not name or name == "." or "/" in name or "\\" in name or ".." in name:
        raise ValueError(
            f"Invalid snapshot name '{name}'. "
            "Names must not be '.', contain path separators, or '..' segments."
        )


class RollbackManager:
    """
    Persist and retrieve LoRA adapter checkpoints keyed by snapshot name.

    All snapshots for a given model live under::

        base_dir / safe_model_name / snapshot_name /
            snapshot.json      ← benchmark scores
            adapter /          ← saved PEFT adapter weights

    By default *base_dir* is ``~/.pyrecall/snapshots``.
    """

    def __init__(
        self,
        model_name: str,
        base_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        root = base_dir if base_dir is not None else Path.home() / ".pyrecall" / "snapshots"

        # IMPORTANT: if base_dir is already a test directory (tmp_snapshot_dir),
        # do NOT nest model_name under a global ~/.pyrecall path
        if base_dir is not None:
            self.base_dir: Path = root / model_name
        else:
            self.base_dir: Path = root / safe_model_name(model_name)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ── saving ─────────────────────────────────────────────────────────────────

    def save(
        self,
        snapshot: SkillSnapshot,
        peft_model: PeftModel,
        compression: str = "none",
    ) -> Path:
        """
        Persist *snapshot* scores and the *peft_model* adapter weights to disk.

        Sets ``snapshot.adapter_path`` before writing so the JSON includes the
        adapter location. Returns the snapshot directory path.

        The adapter is written to a staging directory first. If compression is
        requested, it is applied there before an atomic rename to the final
        ``adapter/`` path, so a crash mid-compression never leaves the snapshot
        in an unloadable half-compressed state.

        Args:
            compression: ``"none"`` (default), ``"gzip"``, ``"zstd"``, or ``"lz4"``.
                         ``"zstd"`` requires ``pip install zstandard``.
                         ``"lz4"`` requires ``pip install lz4``.
        """
        from .compress import compress_adapter_dir

        _validate_snapshot_name(snapshot.name)
        snap_dir = self.base_dir / snapshot.name
        snap_dir.mkdir(parents=True, exist_ok=True)

        with _snap_lock(snap_dir):
            adapter_dir = snap_dir / "adapter"
            adapter_staging = snap_dir / "adapter.staging"

            if adapter_staging.exists():
                shutil.rmtree(adapter_staging)
            adapter_staging.mkdir(parents=True, exist_ok=True)

            peft_model.save_pretrained(str(adapter_staging))
            logger.debug("Adapter staged to %s", adapter_staging)

            if compression != "none":
                compress_adapter_dir(adapter_staging, compression)
                logger.debug("Adapter compressed with %s in staging", compression)

            if adapter_dir.exists():
                shutil.rmtree(adapter_dir)
            adapter_staging.rename(adapter_dir)
            logger.debug("Adapter promoted to %s", adapter_dir)

            snapshot.adapter_path = adapter_dir
            snapshot.adapter_compression = compression
            snapshot.save(snap_dir)
            logger.debug("Snapshot metadata saved to %s", snap_dir)

        return snap_dir

    # ── loading ────────────────────────────────────────────────────────────────

    def load_snapshot(self, name: str) -> SkillSnapshot:
        """
        Load snapshot metadata by name.

        Raises a descriptive error if the snapshot does not exist.
        """
        _validate_snapshot_name(name)
        snap_dir = self.base_dir / name
        if not snap_dir.exists():
            available = self._available_names()
            hint = f" Available snapshots: {available}" if available else " No snapshots saved yet."
            raise FileNotFoundError(f"Snapshot '{name}' not found under '{self.base_dir}'.{hint}")
        with _snap_lock(snap_dir):
            return SkillSnapshot.load(snap_dir)

    def list_snapshots(self) -> list[SkillSnapshot]:
        """Return all saved snapshots sorted by creation time (oldest first).

        Each element is a :class:`~pyrecall.snapshot.SkillSnapshot` object.
        To get just the names use :meth:`list_snapshot_names`.
        """
        if not self.base_dir.exists():
            return []
        snapshots: list[SkillSnapshot] = []
        for snap_dir in sorted(self.base_dir.iterdir()):
            if snap_dir.is_dir() and (snap_dir / "snapshot.json").exists():
                try:
                    with _snap_lock(snap_dir):
                        snapshots.append(SkillSnapshot.load(snap_dir))
                except Exception as exc:
                    logger.warning("Could not load snapshot at %s: %s", snap_dir, exc)
        return sorted(snapshots, key=lambda s: s.created_at)

    def list_snapshot_names(self) -> list[str]:
        """Return the names of all saved snapshots sorted by creation time (oldest first)."""
        return [s.name for s in self.list_snapshots()]

    def delete_snapshot(self, name: str) -> None:
        """Permanently delete a snapshot and its adapter weights."""
        _validate_snapshot_name(name)
        snap_dir = self.base_dir / name
        if not snap_dir.exists():
            raise FileNotFoundError(f"Cannot delete: snapshot '{name}' not found.")
        with _snap_lock(snap_dir):
            shutil.rmtree(snap_dir)
        logger.debug("Deleted snapshot '%s'", name)

    def has_snapshot(self, name: str) -> bool:
        """Return True if *name* refers to a saved snapshot."""
        try:
            _validate_snapshot_name(name)
        except ValueError:
            return False
        return (self.base_dir / name / "snapshot.json").exists()

    # ── private ────────────────────────────────────────────────────────────────

    def _available_names(self) -> list[str]:
        return [
            d.name
            for d in sorted(self.base_dir.iterdir())
            if d.is_dir() and (d / "snapshot.json").exists()
        ]
