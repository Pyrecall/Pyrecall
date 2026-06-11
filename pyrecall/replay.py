"""ReplayBuffer — reservoir-sampled store of past training examples for continual learning."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from .utils import console, get_logger, safe_model_name

logger = get_logger(__name__)

_BUFFER_FILE = "buffer.jsonl"


class ReplayBuffer:
    """
    A fixed-capacity store of past training texts.

    Uses reservoir sampling so every example seen has an equal probability of
    being retained, regardless of insertion order or total volume.  The buffer
    is persisted to ``~/.pyrecall/replay/<model>/buffer.jsonl`` after every
    :meth:`add` call so it survives process restarts.

    Example::

        buf = ReplayBuffer("meta-llama/Llama-3.2-1B", max_size=500)
        buf.add(["example text 1", "example text 2"])
        samples = buf.sample(100)
    """

    def __init__(
        self,
        model_name: str,
        max_size: int = 500,
        base_dir: Optional[Path] = None,
    ) -> None:
        self._max_size = max_size
        self._buffer: list[str] = []
        self._total_seen: int = 0

        root = base_dir or Path.home() / ".pyrecall" / "replay"
        self._path = root / safe_model_name(model_name) / _BUFFER_FILE
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._load()

    # ── public API ─────────────────────────────────────────────────────────────

    def add(self, examples: list[str]) -> None:
        """Add *examples* to the buffer using reservoir sampling."""
        for text in examples:
            self._total_seen += 1
            if len(self._buffer) < self._max_size:
                self._buffer.append(text)
            else:
                j = random.randint(0, self._total_seen - 1)
                if j < self._max_size:
                    self._buffer[j] = text
        self._save()

    def sample(self, n: int) -> list[str]:
        """Return up to *n* randomly sampled examples (without replacement)."""
        if not self._buffer:
            return []
        k = min(n, len(self._buffer))
        return random.sample(self._buffer, k)

    def clear(self) -> None:
        """Empty the buffer and reset counters."""
        self._buffer = []
        self._total_seen = 0
        self._save()

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def total_seen(self) -> int:
        """Total number of examples ever passed to :meth:`add`."""
        return self._total_seen

    # ── persistence ────────────────────────────────────────────────────────────

    def _save(self) -> None:
        meta = {"total_seen": self._total_seen, "max_size": self._max_size}
        lines = [json.dumps(meta)]
        lines += [json.dumps({"text": t}) for t in self._buffer]
        self._path.write_text("\n".join(lines) + "\n")

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            lines = self._path.read_text().splitlines()
            if not lines:
                return
            meta = json.loads(lines[0])
            self._total_seen = meta.get("total_seen", 0)
            self._buffer = [json.loads(l)["text"] for l in lines[1:] if l.strip()]
            # Trim to current max_size in case the config changed.
            if len(self._buffer) > self._max_size:
                self._buffer = self._buffer[: self._max_size]
        except Exception as exc:
            logger.warning("Could not load replay buffer from %s: %s", self._path, exc)
            self._buffer = []
            self._total_seen = 0
