"""SkillSnapshot — a point-in-time record of model capabilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class SkillScore:
    """Score for a single benchmark item."""

    category: str
    prompt: str
    response: str
    score: float  # cosine similarity normalised to [0, 1]

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "prompt": self.prompt,
            "response": self.response,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SkillScore":
        return cls(
            category=data["category"],
            prompt=data["prompt"],
            response=data["response"],
            score=float(data["score"]),
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
    adapter_path: Optional[Path] = None

    # ── aggregation ────────────────────────────────────────────────────────────

    def category_scores(self) -> dict[str, float]:
        """Return average score per category."""
        buckets: dict[str, list[float]] = {}
        for s in self.scores:
            buckets.setdefault(s.category, []).append(s.score)
        return {cat: sum(vals) / len(vals) for cat, vals in buckets.items()}

    def overall_score(self) -> float:
        """Return the mean score across all benchmarks."""
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)

    # ── persistence ────────────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        """Write snapshot metadata to *directory*/snapshot.json."""
        directory.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "model_name": self.model_name,
            "created_at": self.created_at.isoformat(),
            "scores": [s.to_dict() for s in self.scores],
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
        }
        (directory / "snapshot.json").write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, directory: Path) -> "SkillSnapshot":
        """Load a snapshot from *directory*/snapshot.json."""
        snapshot_file = directory / "snapshot.json"
        if not snapshot_file.exists():
            raise FileNotFoundError(
                f"No snapshot.json found in '{directory}'. "
                "Make sure the snapshot was created with model.snapshot()."
            )
        data = json.loads(snapshot_file.read_text())
        return cls(
            name=data["name"],
            model_name=data["model_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            scores=[SkillScore.from_dict(s) for s in data["scores"]],
            adapter_path=Path(data["adapter_path"]) if data["adapter_path"] else None,
        )
