"""SkillSnapshot — a point-in-time record of model capabilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class SkillScore:
    """Score for a single benchmark item."""

    category: str
    prompt: str
    response: str
    score: float  # in [0, 1]; interpretation depends on scoring_method
    scoring_method: str = "log_likelihood"  # "log_likelihood" | "cosine"

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "prompt": self.prompt,
            "response": self.response,
            "score": self.score,
            "scoring_method": self.scoring_method,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SkillScore:
        return cls(
            category=data["category"],
            prompt=data["prompt"],
            response=data["response"],
            score=float(data["score"]),
            # default "cosine" preserves backward compat with old snapshots
            scoring_method=data.get("scoring_method", "cosine"),
        )


@dataclass
class SkillSnapshot:
    """
    A named snapshot of model skill scores.

    Stores benchmark responses + scores and optionally a path to the saved
    LoRA adapter so the model can be rolled back to this exact state.
    """

    name: str
    model_name: str
    created_at: datetime = field(default_factory=datetime.now)
    scores: list[SkillScore] = field(default_factory=list)
    adapter_path: Path | None = None
    encrypted: bool = False
    adapter_compression: str = "none"

    def __post_init__(self) -> None:
        if not isinstance(self.created_at, datetime):
            raise TypeError(
                f"created_at must be a datetime, got {type(self.created_at).__name__}. "
                "Use keyword arguments: SkillSnapshot(name=..., model_name=..., scores=...)"
            )

    # ── aggregation ────────────────────────────────────────────────────────────
    def category_scores(self) -> dict[str, float]:
        """Return average score per category."""
        buckets: dict[str, list[float]] = {}
        for s in self.scores:
            buckets.setdefault(s.category, []).append(s.score)
        return {cat: sum(vals) / len(vals) for cat, vals in buckets.items()}

    def overall_score(self) -> float:
        """Return the mean of per-category averages, so every skill is weighted equally."""
        cat_scores = self.category_scores()
        if not cat_scores:
            return 0.0
        return sum(cat_scores.values()) / len(cat_scores)

    def primary_scoring_method(self) -> str | None:
        """Return the most common ``scoring_method`` across all scores.

        Used (instead of comparing the full set of per-score methods) so a
        snapshot loaded with a handful of legacy scores defaulting to
        ``"cosine"`` (see :meth:`SkillScore.from_dict`) isn't flagged as
        mismatched against a snapshot that is overwhelmingly one method.
        """
        if not self.scores:
            return None
        counts: dict[str, int] = {}
        for s in self.scores:
            counts[s.scoring_method] = counts.get(s.scoring_method, 0) + 1
        return max(counts, key=lambda k: counts[k])

    # ── persistence ────────────────────────────────────────────────────────────
    def save(self, directory: Path, privacy: bool = False, passphrase: str | None = None) -> None:
        """Write snapshot metadata to ``directory/snapshot.json``.

        The file is always named ``snapshot.json`` inside *directory* —
        no subdirectory is created from the snapshot name.  Pass the
        snapshot's own directory (e.g. ``base / snapshot.name``) if you
        want the canonical on-disk layout used by :class:`RollbackManager`.

        When *privacy* is ``True`` a *passphrase* must be supplied.  The key
        is derived from the passphrase via PBKDF2-HMAC-SHA256; only the salt
        (safe to store publicly) is written to disk — the key itself never is.
        """
        directory.mkdir(parents=True, exist_ok=True)
        if not privacy:
            data = {
                "encrypted": False,
                "name": self.name,
                "model_name": self.model_name,
                "created_at": self.created_at.isoformat(),
                "scores": [s.to_dict() for s in self.scores],
                "adapter_path": str(self.adapter_path) if self.adapter_path else None,
                "adapter_compression": self.adapter_compression,
            }
        else:
            if not passphrase:
                raise ValueError(
                    "privacy=True requires a passphrase. "
                    "Pass passphrase='your-secret' to snapshot.save()."
                )
            from .encrypt import Encryptor

            encryptor = Encryptor.from_passphrase(passphrase)
            import base64
            data = {
                "encrypted": True,
                "salt": base64.b64encode(encryptor.salt).decode(),
                "name": encryptor.encrypt(self.name),
                "model_name": encryptor.encrypt(self.model_name),
                "created_at": encryptor.encrypt(self.created_at.isoformat()),
                "scores": encryptor.encrypt(json.dumps([s.to_dict() for s in self.scores])),
                "adapter_path": (
                    encryptor.encrypt(str(self.adapter_path)) if self.adapter_path else None
                ),
                "adapter_compression": self.adapter_compression,
            }
        (directory / "snapshot.json").write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, directory: Path, privacy: bool = False, passphrase: str | None = None) -> "SkillSnapshot":
        """Load a snapshot from *directory*/snapshot.json.

        When *privacy* is ``True`` the same *passphrase* used during
        :meth:`save` must be supplied so the key can be re-derived from the
        stored salt.
        """
        snapshot_file = directory / "snapshot.json"
        if not snapshot_file.exists():
            raise FileNotFoundError(
                f"No snapshot.json found in '{directory}'. "
                "Make sure the snapshot was created with model.snapshot()."
            )
        try:
            data = json.loads(snapshot_file.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Snapshot file '{snapshot_file}' is corrupted (invalid JSON): {exc}"
            ) from exc
        is_encrypted = data.get("encrypted", False)
        if is_encrypted and not privacy:
            raise ValueError(
                f"Snapshot '{snapshot_file}' is encrypted but privacy=False was passed. "
                "Load it with privacy=True and supply the passphrase."
            )
        if privacy and not is_encrypted:
            raise ValueError(
                f"Snapshot '{snapshot_file}' is not encrypted but privacy=True was passed. "
                "Load it with privacy=False."
            )
        if privacy:
            if not passphrase:
                raise ValueError(
                    "privacy=True requires a passphrase. "
                    "Pass the same passphrase used when saving the snapshot."
                )
            import base64
            from .encrypt import Encryptor

            salt = base64.b64decode(data["salt"])
            encryptor = Encryptor.from_passphrase(passphrase, salt=salt)
            return cls(
                name=encryptor.decrypt(data["name"]),
                model_name=encryptor.decrypt(data["model_name"]),
                created_at=datetime.fromisoformat(encryptor.decrypt(data["created_at"])),
                scores=[
                    SkillScore.from_dict(s) for s in json.loads(encryptor.decrypt(data["scores"]))
                ],
                adapter_path=(
                    Path(encryptor.decrypt(data["adapter_path"])) if data["adapter_path"] else None
                ),
                adapter_compression=data.get("adapter_compression", "none"),
                encrypted=True,
            )
        return cls(
            name=data["name"],
            model_name=data["model_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            scores=[SkillScore.from_dict(s) for s in data["scores"]],
            adapter_path=Path(data["adapter_path"]) if data["adapter_path"] else None,
            adapter_compression=data.get("adapter_compression", "none"),
        )
