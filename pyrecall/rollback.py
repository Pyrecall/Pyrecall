"""RollbackManager — save and restore LoRA adapter checkpoints."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from .snapshot import SkillSnapshot
from .utils import get_logger, safe_model_name

if TYPE_CHECKING:
    from peft import PeftModel

logger = get_logger(__name__)


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
        self.base_dir: Path = (
            base_dir
            if base_dir is not None
            else Path.home() / ".pyrecall" / "snapshots" / safe_model_name(model_name)
        )
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

        Args:
            compression: ``"none"`` (default), ``"gzip"``, ``"zstd"``, or ``"lz4"``.
                         ``"zstd"`` requires ``pip install zstandard``.
                         ``"lz4"`` requires ``pip install lz4``.
        """
        from .compress import compress_adapter_dir

        snap_dir = self.base_dir / snapshot.name
        snap_dir.mkdir(parents=True, exist_ok=True)

        adapter_dir = snap_dir / "adapter"
        peft_model.save_pretrained(str(adapter_dir))
        logger.debug("Adapter saved to %s", adapter_dir)

        if compression != "none":
            compress_adapter_dir(adapter_dir, compression)
            logger.debug("Adapter compressed with %s", compression)

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
        snap_dir = self.base_dir / name
        if not snap_dir.exists():
            available = self._available_names()
            hint = f" Available snapshots: {available}" if available else " No snapshots saved yet."
            raise FileNotFoundError(f"Snapshot '{name}' not found under '{self.base_dir}'.{hint}")
        return SkillSnapshot.load(snap_dir)

    def list_snapshots(self) -> list[SkillSnapshot]:
        """Return all saved snapshots sorted by creation time (oldest first)."""
        if not self.base_dir.exists():
            return []
        snapshots: list[SkillSnapshot] = []
        for snap_dir in sorted(self.base_dir.iterdir()):
            if snap_dir.is_dir() and (snap_dir / "snapshot.json").exists():
                try:
                    snapshots.append(SkillSnapshot.load(snap_dir))
                except Exception as exc:
                    logger.warning("Could not load snapshot at %s: %s", snap_dir, exc)
        return sorted(snapshots, key=lambda s: s.created_at)

    def delete_snapshot(self, name: str) -> None:
        """Permanently delete a snapshot and its adapter weights."""
        snap_dir = self.base_dir / name
        if not snap_dir.exists():
            raise FileNotFoundError(f"Cannot delete: snapshot '{name}' not found.")
        shutil.rmtree(snap_dir)
        logger.debug("Deleted snapshot '%s'", name)

    def has_snapshot(self, name: str) -> bool:
        """Return True if *name* refers to a saved snapshot."""
        return (self.base_dir / name / "snapshot.json").exists()

    # ── private ────────────────────────────────────────────────────────────────

    def _available_names(self) -> list[str]:
        return [
            d.name
            for d in sorted(self.base_dir.iterdir())
            if d.is_dir() and (d / "snapshot.json").exists()
        ]
