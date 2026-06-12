"""Custom benchmark management — load, register, and remove user-defined prompt suites."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .default import Benchmark

_DEFAULT_BENCHMARK_DIR = Path.home() / ".pyrecall" / "benchmarks"


class CustomBenchmarkManager:
    """
    Manages user-defined benchmark suites stored under *base_dir*.

    Each suite is a ``.jsonl`` file where every line is a JSON object with
    at least ``"prompt"`` and ``"reference_answer"`` keys.  An optional
    ``"category"`` key labels the skill being tested; if omitted the suite
    filename (without extension) is used as the category name.

    Example file ``nautical.jsonl``::

        {"prompt": "What does 'port' mean on a ship?", "reference_answer": "The left side when facing the bow.", "category": "nautical"}
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or _DEFAULT_BENCHMARK_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ─────────────────────────────────────────────────────────────

    def add(self, path: str | Path, name: str | None = None) -> str:
        """
        Register a JSONL benchmark file.

        Copies *path* into the benchmark store.  Raises ``ValueError`` if the
        file is not valid JSONL or is missing required keys.

        Args:
            path: Path to the source ``.jsonl`` file.
            name: Name for the suite (used as filename and default category).
                  Defaults to the source file's stem.

        Returns:
            The registered suite name.
        """
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Benchmark file not found: '{src}'")

        entries = _parse_jsonl(src)
        if not entries:
            raise ValueError(f"'{src}' contains no valid benchmark entries.")

        suite_name = name or src.stem
        dest = self.base_dir / f"{suite_name}.jsonl"
        shutil.copy2(src, dest)
        return suite_name

    def suites(self) -> list[dict]:
        """
        Return metadata for every registered suite.

        Returns:
            List of dicts with keys ``name``, ``path``, and ``count``.
        """
        result = []
        for f in sorted(self.base_dir.glob("*.jsonl")):
            entries = _parse_jsonl(f)
            result.append({"name": f.stem, "path": str(f), "count": len(entries)})
        return result

    def remove(self, name: str) -> None:
        """
        Delete a registered suite by name.

        Raises ``FileNotFoundError`` if no suite with that name exists.
        """
        target = self.base_dir / f"{name}.jsonl"
        if not target.exists():
            raise FileNotFoundError(f"No benchmark suite named '{name}'.")
        target.unlink()

    def load_all(self) -> list[Benchmark]:
        """
        Load every registered suite and return a flat list of :class:`Benchmark` objects.
        """
        benchmarks: list[Benchmark] = []
        for f in sorted(self.base_dir.glob("*.jsonl")):
            default_category = f.stem
            for entry in _parse_jsonl(f):
                benchmarks.append(
                    Benchmark(
                        category=entry.get("category", default_category),
                        prompt=entry["prompt"],
                        reference_answer=entry["reference_answer"],
                    )
                )
        return benchmarks

    def count(self) -> int:
        """Total number of custom benchmark prompts across all suites."""
        return sum(info["count"] for info in self.suites())


# ── helpers ───────────────────────────────────────────────────────────────────


def _parse_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file and return only entries that have the required keys."""
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "prompt" in obj and "reference_answer" in obj:
            entries.append(obj)
    return entries
