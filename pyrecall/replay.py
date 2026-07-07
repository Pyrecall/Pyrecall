"""ReplayBuffer — reservoir-sampled store of past training examples for continual learning."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

from .utils import get_logger, safe_model_name

logger = get_logger(__name__)

_BUFFER_FILE = "buffer.jsonl"


def _weighted_sample_without_replacement(
    pool: list[dict], weights: list[float], k: int
) -> list[str]:
    """Return k text strings sampled from pool proportional to weights, without replacement."""
    pool = list(pool)
    weights = list(weights)
    result: list[str] = []
    for _ in range(k):
        total = sum(weights)
        if total <= 0:
            remaining = k - len(result)
            if pool:
                result.extend(e["text"] for e in random.sample(pool, min(remaining, len(pool))))
            break
        r = random.uniform(0, total)
        cumulative = 0.0
        for i, (entry, w) in enumerate(zip(pool, weights)):
            cumulative += w
            if cumulative >= r:
                result.append(entry["text"])
                pool.pop(i)
                weights.pop(i)
                break
        else:
            # Floating-point edge case: r landed exactly on total but cumulative
            # fell fractionally short on the last step.  Pick the last entry.
            result.append(pool[-1]["text"])
            pool.pop()
            weights.pop()
    return result


class ReplayBuffer:
    """
    A fixed-capacity store of past training texts.

    Uses reservoir sampling so every example seen has an equal probability of
    being retained, regardless of insertion order or total volume.  The buffer
    is persisted to ``~/.pyrecall/replay/<model>/buffer.jsonl`` after every
    :meth:`add` call so it survives process restarts.

    Each entry optionally carries a ``category`` label (taken from the JSONL
    ``"category"`` field if present).  This enables weighted replay via
    :meth:`sample` so forgotten categories can be up-weighted on the next
    training run.

    Example::

        buf = ReplayBuffer("meta-llama/Llama-3.2-1B", max_size=500)
        buf.add(["example text 1", "example text 2"])
        samples = buf.sample(100)
    """

    def __init__(
        self,
        model_name: str,
        max_size: int = 500,
        base_dir: Path | None = None,
    ) -> None:
        self._max_size = max_size
        self._buffer: list[dict] = []  # each entry: {"text": str, "category": str | None}
        self._total_seen: int = 0
        self._seen_hashes: set[str] = set()

        root = base_dir or Path.home() / ".pyrecall" / "replay"
        self._path = root / safe_model_name(model_name) / _BUFFER_FILE
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._load()

    # ── public API ─────────────────────────────────────────────────────────────

    def add(self, examples: list[str], categories: list[str | None] | None = None) -> None:
        """Add *examples* to the buffer using reservoir sampling, skipping duplicates.

        Args:
            examples: Training text strings to add.
            categories: Optional category label for each example.  Must be the
                same length as *examples* when provided.  Entries without a
                category are stored with ``None`` and treated as uncategorised
                during weighted sampling.
        """
        if categories is None:
            categories = [None] * len(examples)

        duplicates = 0
        for text, cat in zip(examples, categories):
            h = hashlib.sha256(text.encode()).hexdigest()
            if h in self._seen_hashes:
                duplicates += 1
                continue
            self._seen_hashes.add(h)
            self._total_seen += 1
            entry = {"text": text, "category": cat}
            if len(self._buffer) < self._max_size:
                self._buffer.append(entry)
            else:
                j = random.randint(0, self._total_seen - 1)
                if j < self._max_size:
                    self._buffer[j] = entry
        if duplicates:
            logger.debug(
                "ReplayBuffer.add(): skipped %d duplicate example(s) "
                "(same data seen before — replay_mix_ratio reflects unique examples only).",
                duplicates,
            )
        self._save()

    def sample(self, n: int, weights: dict[str, float] | None = None) -> list[str]:
        """Return up to *n* sampled examples (without replacement).

        Args:
            n: Maximum number of examples to return.
            weights: Optional mapping of category → positive multiplier.  When
                provided, examples whose category has a higher weight are more
                likely to be selected.  Examples with no category are treated
                as weight ``1.0``.  Falls back to uniform sampling when
                ``weights`` is ``None`` or the buffer has no category metadata.
        """
        if not self._buffer:
            return []
        k = min(n, len(self._buffer))

        if weights is None:
            return [e["text"] for e in random.sample(self._buffer, k)]

        entry_weights = [max(0.0, weights.get(e.get("category"), 1.0)) for e in self._buffer]
        return _weighted_sample_without_replacement(self._buffer, entry_weights, k)

    def clear(self) -> None:
        """Empty the buffer and reset counters."""
        self._buffer = []
        self._total_seen = 0
        self._seen_hashes = set()
        self._save()

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def total_seen(self) -> int:
        """Total number of *unique* examples ever accepted by :meth:`add`.

        Duplicates are excluded intentionally: this counter drives reservoir
        sampling probability, so counting only unique texts keeps the sampling
        distribution honest even when the same file is learned more than once.
        """
        return self._total_seen

    # ── persistence ────────────────────────────────────────────────────────────

    def _save(self) -> None:
        meta = {
            "total_seen": self._total_seen,
            "max_size": self._max_size,
            "seen_hashes": list(self._seen_hashes),
        }
        lines = [json.dumps(meta)]
        lines += [
            json.dumps({"text": e["text"], "category": e.get("category")}) for e in self._buffer
        ]
        content = "\n".join(lines) + "\n"
        tmp_fd, tmp_path = tempfile.mkstemp(dir=self._path.parent, prefix=".buffer_tmp_")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.write(content)
                fh.flush()
                os.fsync(fh.fileno())

            os.replace(tmp_path, self._path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        finally:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            lines = self._path.read_text().splitlines()
            if not lines:
                return
            meta = json.loads(lines[0])

            required = {"total_seen", "max_size"}
            missing = required - meta.keys()
            if missing:
                raise ValueError(
                    f"Replay buffer metadata missing field(s): {', '.join(sorted(missing))}"
                )

            if not isinstance(meta["total_seen"], int):
                raise ValueError("Replay buffer 'total_seen' must be an integer.")

            if not isinstance(meta["max_size"], int):
                raise ValueError("Replay buffer 'max_size' must be an integer.")

            if not isinstance(meta["seen_hashes"], list):
                raise ValueError("Replay buffer 'seen_hashes' must be a list.")

            self._total_seen = meta["total_seen"]
            self._seen_hashes = set(meta["seen_hashes"])
            self._buffer = []
            skipped = 0
            for line in lines[1:]:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if "text" not in entry:
                        raise KeyError("text")
                    # backward compat: old format only had "text", no "category"
                    if "category" not in entry:
                        entry["category"] = None
                    self._buffer.append(entry)
                except Exception as exc:
                    skipped += 1
                    logger.warning("Skipping corrupt replay buffer entry: %s", exc)
            if skipped:
                logger.warning(
                    "Skipped %d corrupt entr%s while loading %s.",
                    skipped,
                    "y" if skipped == 1 else "ies",
                    self._path,
                )
            # Trim to current max_size in case the config changed.
            # Use random.sample to preserve reservoir-sampling distribution.
            if len(self._buffer) > self._max_size:
                self._buffer = random.sample(self._buffer, self._max_size)
            # Backward compat: old files without seen_hashes in meta fall back to
            # rebuilding from the buffer (misses evicted entries, but better than nothing).
            if "seen_hashes" not in meta:
                self._seen_hashes = {
                    hashlib.sha256(e["text"].encode()).hexdigest() for e in self._buffer
                }
        except Exception as exc:
            logger.warning("Could not load replay buffer from %s: %s", self._path, exc)
            self._buffer = []
            self._total_seen = 0
            self._seen_hashes = set()
