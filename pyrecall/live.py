"""LiveLearner — collect production interactions and trigger periodic fine-tuning."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .utils import console, get_logger

if TYPE_CHECKING:
    from .model import Model

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt      TEXT    NOT NULL,
    response    TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    trained     INTEGER NOT NULL DEFAULT 0
);
"""


class LiveLearner:
    """
    Collect (prompt, response) pairs from a live server and periodically
    fine-tune the model on that data.

    How it works:

    1. Each call to :meth:`record` appends an interaction to a local SQLite DB.
    2. Once *batch_size* untrained interactions accumulate, a JSONL file is
       exported and :py:meth:`Model.learn` is called for one epoch.
    3. Trained rows are marked so they are never included in a future batch.

    The SQLite database lives at ``~/.pyrecall/live_data.db`` by default.
    """

    def __init__(
        self,
        model: Model,
        batch_size: int = 50,
        db_path: Path | None = None,
        min_response_length: int = 10,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.db_path: Path = db_path or Path.home() / ".pyrecall" / "live_data.db"
        self.min_response_length = min_response_length
        self._training_lock = threading.Lock()
        self._training_thread: threading.Thread | None = None
        self._init_db()

    # ── public API ─────────────────────────────────────────────────────────────

    def record(self, prompt: str, response: str) -> None:
        """
        Store one interaction.

        Silently skips responses shorter than *min_response_length* (likely
        error messages or empty outputs that would degrade training).
        Triggers a training run if the untrained batch is full.
        """
        if len(response.strip()) < self.min_response_length:
            logger.debug(
                "record() skipped: response too short (%d chars < min %d)",
                len(response.strip()),
                self.min_response_length,
            )
            return

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO interactions (prompt, response, timestamp) VALUES (?, ?, ?)",
                (prompt, response, datetime.now().isoformat()),
            )

        self._maybe_trigger_training()

    def pending_count(self) -> int:
        """Number of interactions not yet used for training."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM interactions WHERE trained = 0").fetchone()
        return row[0]

    def total_count(self) -> int:
        """Total number of recorded interactions (trained + pending)."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()
        return row[0]

    def clear_pending(self) -> None:
        """Delete all untrained interactions (does not affect trained rows)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM interactions WHERE trained = 0")

    # ── private ────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _join(self) -> None:
        """Wait for any in-progress background training thread to finish."""
        if self._training_thread is not None:
            self._training_thread.join()

    def _maybe_trigger_training(self) -> None:
        if self.pending_count() >= self.batch_size:
            if self._training_lock.acquire(blocking=False):
                self._training_thread = threading.Thread(
                    target=self._trigger_training_locked, daemon=True
                )
                self._training_thread.start()

    def _trigger_training_locked(self) -> None:
        try:
            self._trigger_training()
        finally:
            self._training_lock.release()

    def _trigger_training(self) -> None:
        """Export the current pending batch to JSONL and call model.learn()."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, prompt, response FROM interactions "
                "WHERE trained = 0 ORDER BY id LIMIT ?",
                (self.batch_size,),
            ).fetchall()

        if not rows:
            return

        console.print(f"[info]LiveLearner: training on {len(rows)} new interactions…[/info]")

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
                for row in rows:
                    entry = {
                        "text": (f"### Human: {row['prompt']}\n\n### Assistant: {row['response']}")
                    }
                    fh.write(json.dumps(entry) + "\n")
                tmp_path = Path(fh.name)

            self.model.learn(str(tmp_path), epochs=1)

            row_ids = [(row["id"],) for row in rows]
            with self._connect() as conn:
                conn.executemany("UPDATE interactions SET trained = 1 WHERE id = ?", row_ids)

            console.print(
                f"[success]✓ LiveLearner: fine-tuned on {len(rows)} interactions.[/success]"
            )
        except Exception as exc:
            logger.error("LiveLearner training run failed: %s", exc)
            console.print(f"[error]LiveLearner training failed: {exc}[/error]")
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
