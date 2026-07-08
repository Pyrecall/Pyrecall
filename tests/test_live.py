"""Tests for LiveLearner — SQLite logging, file export, batch trigger logic, edge cases."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyrecall.live import LiveLearner

# ── helpers ────────────────────────────────────────────────────────────────────


def _make_learner(
    tmp_path: Path,
    batch_size: int = 3,
    min_response_length: int = 5,
) -> tuple[LiveLearner, MagicMock]:
    """Return a LiveLearner backed by a temp DB and a mocked Model."""
    model = MagicMock()
    # Add a mock RLock for thread-safety testing
    model._model_lock = MagicMock()
    model._model_lock.acquire.return_value = True
    model._model_lock.release.return_value = None
    db_path = tmp_path / "live_data.db"
    learner = LiveLearner(
        model=model,
        batch_size=batch_size,
        db_path=db_path,
        min_response_length=min_response_length,
    )
    return learner, model


def _record_and_join(learner: LiveLearner, prompt: str, response: str) -> None:
    """Record an interaction and wait for any triggered training to finish."""
    learner.record(prompt, response)
    learner._join()


# ── SQLite logging ─────────────────────────────────────────────────────────────


class TestSQLiteLogging:
    def test_db_file_created_on_init(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path)
        assert learner.db_path.exists()

    def test_table_schema_exists(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path)
        conn = sqlite3.connect(learner.db_path)
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        conn.close()
        assert "interactions" in tables

    def test_pending_count_zero_for_empty_db(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path)
        assert learner.pending_count() == 0

    def test_total_count_zero_for_empty_db(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path)
        assert learner.total_count() == 0

    def test_record_increments_pending_count(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=100)
        learner.record("hello", "world response text")
        assert learner.pending_count() == 1

    def test_record_increments_total_count(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=100)
        learner.record("hello", "world response text")
        assert learner.total_count() == 1

    def test_multiple_records_accumulate(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=100)
        for i in range(5):
            learner.record(f"prompt {i}", f"response {i} long enough text here")
        assert learner.pending_count() == 5
        assert learner.total_count() == 5

    def test_duplicate_entries_both_stored(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=100)
        learner.record("same prompt", "same response long text")
        learner.record("same prompt", "same response long text")
        assert learner.total_count() == 2

    def test_record_stores_correct_prompt(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=100)
        learner.record("my unique prompt", "my long response text here")
        conn = sqlite3.connect(learner.db_path)
        row = conn.execute("SELECT prompt FROM interactions").fetchone()
        conn.close()
        assert row[0] == "my unique prompt"

    def test_record_stores_correct_response(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=100)
        learner.record("prompt", "my long response text here")
        conn = sqlite3.connect(learner.db_path)
        row = conn.execute("SELECT response FROM interactions").fetchone()
        conn.close()
        assert row[0] == "my long response text here"

    def test_record_stores_timestamp(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=100)
        learner.record("p", "long response text here extra")
        conn = sqlite3.connect(learner.db_path)
        row = conn.execute("SELECT timestamp FROM interactions").fetchone()
        conn.close()
        assert row[0] is not None and len(row[0]) > 0

    def test_record_stores_trained_as_zero(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=100)
        learner.record("p", "long response text here extra")
        conn = sqlite3.connect(learner.db_path)
        row = conn.execute("SELECT trained FROM interactions").fetchone()
        conn.close()
        assert row[0] == 0

    def test_clear_pending_removes_untrained_rows(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=100)
        learner.record("p1", "response one long text")
        learner.record("p2", "response two long text")
        learner.clear_pending()
        assert learner.pending_count() == 0
        assert learner.total_count() == 0

    def test_clear_pending_preserves_trained_rows(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=2)
        _record_and_join(learner, "p1", "response one long enough text")
        _record_and_join(learner, "p2", "response two long enough text")
        # Both rows are now trained after the batch trigger
        trained_count = learner.total_count() - learner.pending_count()
        assert trained_count == 2
        learner.clear_pending()
        assert learner.total_count() == trained_count

    def test_new_learner_with_existing_db_reads_correctly(self, tmp_path: Path) -> None:
        learner1, _ = _make_learner(tmp_path, batch_size=100)
        learner1.record("p", "some long response text here")
        # Create a second learner pointing at the same DB
        model2 = MagicMock()
        learner2 = LiveLearner(model=model2, batch_size=100, db_path=learner1.db_path)
        assert learner2.total_count() == 1
        assert learner2.pending_count() == 1


# ── short response filtering ──────────────────────────────────────────────────


class TestShortResponseFiltering:
    def test_response_below_min_length_not_stored(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=100, min_response_length=10)
        learner.record("prompt", "short")  # 5 chars < 10
        assert learner.total_count() == 0

    def test_response_at_exactly_min_length_is_stored(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=100, min_response_length=5)
        learner.record("prompt", "hello")  # exactly 5 chars after strip
        assert learner.total_count() == 1

    def test_response_above_min_length_is_stored(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=100, min_response_length=3)
        learner.record("prompt", "long enough response")
        assert learner.total_count() == 1

    def test_whitespace_only_response_is_skipped(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=100, min_response_length=1)
        learner.record("prompt", "   \n\t  ")
        assert learner.total_count() == 0

    def test_empty_response_is_skipped(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=100, min_response_length=1)
        learner.record("prompt", "")
        assert learner.total_count() == 0

    def test_length_check_uses_stripped_text(self, tmp_path: Path) -> None:
        # "  x  " is 5 chars raw but 1 char stripped — should be skipped when min=3
        learner, _ = _make_learner(tmp_path, batch_size=100, min_response_length=3)
        learner.record("prompt", "  x  ")
        assert learner.total_count() == 0

    def test_training_not_triggered_if_all_responses_too_short(self, tmp_path: Path) -> None:
        learner, model = _make_learner(tmp_path, batch_size=2, min_response_length=50)
        for _ in range(5):
            learner.record("prompt", "short")
        model.learn.assert_not_called()


# ── batch trigger logic ───────────────────────────────────────────────────────


class TestBatchTriggerLogic:
    def test_training_not_triggered_below_batch_size(self, tmp_path: Path) -> None:
        learner, model = _make_learner(tmp_path, batch_size=3)
        learner.record("p1", "response one long enough text")
        learner.record("p2", "response two long enough text")
        model.learn.assert_not_called()

    def test_training_triggered_at_exactly_batch_size(self, tmp_path: Path) -> None:
        learner, model = _make_learner(tmp_path, batch_size=3)
        for i in range(3):
            _record_and_join(learner, f"p{i}", f"response {i} long enough text here")
        model.learn.assert_called_once()

    def test_training_triggered_again_after_second_full_batch(self, tmp_path: Path) -> None:
        learner, model = _make_learner(tmp_path, batch_size=2)
        for i in range(4):
            _record_and_join(learner, f"p{i}", f"response {i} long enough text here")
        assert model.learn.call_count == 2

    def test_rows_marked_trained_after_batch(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=2)
        _record_and_join(learner, "p1", "response one long enough text")
        _record_and_join(learner, "p2", "response two long enough text")
        assert learner.pending_count() == 0

    def test_total_count_preserved_after_training(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=2)
        _record_and_join(learner, "p1", "response one long enough text")
        _record_and_join(learner, "p2", "response two long enough text")
        assert learner.total_count() == 2

    def test_pending_count_after_partial_second_batch(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=3)
        for i in range(4):
            _record_and_join(learner, f"p{i}", f"response {i} long enough text here")
        # 3 trained, 1 pending
        assert learner.pending_count() == 1
        assert learner.total_count() == 4

    def test_only_batch_size_rows_trained_per_trigger(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=3)
        for i in range(5):
            _record_and_join(learner, f"p{i}", f"response {i} long enough text here")
        # Only 3 rows trained, 2 still pending
        assert learner.pending_count() == 2


# ── file export ────────────────────────────────────────────────────────────────


class TestFileExport:
    def test_learn_called_with_a_jsonl_path(self, tmp_path: Path) -> None:
        learner, model = _make_learner(tmp_path, batch_size=2)
        _record_and_join(learner, "p1", "response one long text here")
        _record_and_join(learner, "p2", "response two long text here")
        model.learn.assert_called_once()
        call_path: str = model.learn.call_args[0][0]
        assert call_path.endswith(".jsonl")

    def test_learn_called_with_epochs_1(self, tmp_path: Path) -> None:
        learner, model = _make_learner(tmp_path, batch_size=2)
        _record_and_join(learner, "p1", "response one long text here")
        _record_and_join(learner, "p2", "response two long text here")
        assert model.learn.call_args[1]["epochs"] == 1

    def test_exported_jsonl_line_count_matches_batch(self, tmp_path: Path) -> None:
        captured: list[list[str]] = []

        def capture(path: str, epochs: int) -> None:  # noqa: ARG001
            with open(path) as fh:
                captured.append(fh.readlines())

        learner, model = _make_learner(tmp_path, batch_size=3)
        model.learn.side_effect = capture
        for i in range(3):
            _record_and_join(learner, f"prompt {i}", f"response {i} long enough text here")

        assert len(captured) == 1
        assert len(captured[0]) == 3

    def test_exported_jsonl_has_text_key(self, tmp_path: Path) -> None:
        captured_lines: list[str] = []

        def capture(path: str, epochs: int) -> None:  # noqa: ARG001
            with open(path) as fh:
                captured_lines.extend(fh.readlines())

        learner, model = _make_learner(tmp_path, batch_size=2)
        model.learn.side_effect = capture
        _record_and_join(learner, "hello world prompt", "this is a response text")
        _record_and_join(learner, "second prompt text", "another response here text")

        entry = json.loads(captured_lines[0])
        assert "text" in entry

    def test_exported_jsonl_contains_prompt_and_response(self, tmp_path: Path) -> None:
        captured_texts: list[str] = []

        def capture(path: str, epochs: int) -> None:  # noqa: ARG001
            with open(path) as fh:
                for line in fh:
                    captured_texts.append(json.loads(line)["text"])

        learner, model = _make_learner(tmp_path, batch_size=1, min_response_length=1)
        model.learn.side_effect = capture
        _record_and_join(learner, "My question here", "My answer here")

        assert len(captured_texts) == 1
        assert "My question here" in captured_texts[0]
        assert "My answer here" in captured_texts[0]

    def test_exported_jsonl_uses_human_assistant_format(self, tmp_path: Path) -> None:
        captured_texts: list[str] = []

        def capture(path: str, epochs: int) -> None:  # noqa: ARG001
            with open(path) as fh:
                for line in fh:
                    captured_texts.append(json.loads(line)["text"])

        learner, model = _make_learner(tmp_path, batch_size=1, min_response_length=1)
        model.learn.side_effect = capture
        _record_and_join(learner, "question", "answer")

        assert "### Human:" in captured_texts[0]
        assert "### Assistant:" in captured_texts[0]

    def test_temp_file_deleted_after_successful_training(self, tmp_path: Path) -> None:
        saved_paths: list[str] = []

        def capture(path: str, epochs: int) -> None:  # noqa: ARG001
            saved_paths.append(path)

        learner, model = _make_learner(tmp_path, batch_size=1, min_response_length=1)
        model.learn.side_effect = capture
        _record_and_join(learner, "p", "a")

        assert len(saved_paths) == 1
        assert not Path(saved_paths[0]).exists()


# ── failed training ────────────────────────────────────────────────────────────


class TestFailedTraining:
    def test_rows_not_marked_trained_after_failure(self, tmp_path: Path) -> None:
        learner, model = _make_learner(tmp_path, batch_size=2)
        model.learn.side_effect = RuntimeError("GPU out of memory")
        _record_and_join(learner, "p1", "response one long text here")
        _record_and_join(learner, "p2", "response two long text here")
        # Training failed — rows must still be pending
        assert learner.pending_count() == 2

    def test_total_count_unchanged_after_failure(self, tmp_path: Path) -> None:
        learner, model = _make_learner(tmp_path, batch_size=2)
        model.learn.side_effect = RuntimeError("OOM")
        _record_and_join(learner, "p1", "response one long text here")
        _record_and_join(learner, "p2", "response two long text here")
        assert learner.total_count() == 2

    def test_db_still_queryable_after_failure(self, tmp_path: Path) -> None:
        learner, model = _make_learner(tmp_path, batch_size=2)
        model.learn.side_effect = RuntimeError("OOM")
        _record_and_join(learner, "p1", "response one long text here")
        _record_and_join(learner, "p2", "response two long text here")
        # Should not raise
        assert learner.pending_count() >= 0

    def test_temp_file_deleted_even_after_failure(self, tmp_path: Path) -> None:
        saved_paths: list[str] = []

        def fail_and_record(path: str, epochs: int) -> None:  # noqa: ARG001
            saved_paths.append(path)
            raise RuntimeError("training failed")

        learner, model = _make_learner(tmp_path, batch_size=1, min_response_length=1)
        model.learn.side_effect = fail_and_record
        _record_and_join(learner, "p", "a")

        assert len(saved_paths) == 1
        assert not Path(saved_paths[0]).exists()

    def test_subsequent_records_still_work_after_failure(self, tmp_path: Path) -> None:
        learner, model = _make_learner(tmp_path, batch_size=2)
        # First training fails, second succeeds.
        model.learn.side_effect = [RuntimeError("fail"), None]
        # Two records → first training fires and fails; pending rows stay at 2.
        _record_and_join(learner, "p0", "response 0 long text here extra")
        _record_and_join(learner, "p1", "response 1 long text here extra")
        # Lock is now released; third record → pending becomes 3 → second training fires.
        _record_and_join(learner, "p2", "response 2 long text here extra")
        assert model.learn.call_count == 2
        # Two rows were marked trained on the second run.
        assert learner.total_count() - learner.pending_count() == 2


# ── edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_trigger_training_on_empty_pending_is_no_op(self, tmp_path: Path) -> None:
        learner, model = _make_learner(tmp_path, batch_size=2)
        # Call _trigger_training directly when there are no pending rows
        learner._trigger_training()
        model.learn.assert_not_called()

    def test_pending_count_only_counts_untrained(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=2)
        # Fill one batch (marks 2 as trained)
        _record_and_join(learner, "p1", "response one long text here")
        _record_and_join(learner, "p2", "response two long text here")
        # Add one more pending row
        learner.record("p3", "response three long text here")
        assert learner.pending_count() == 1
        assert learner.total_count() == 3

    def test_large_batch_size_never_triggers(self, tmp_path: Path) -> None:
        learner, model = _make_learner(tmp_path, batch_size=1000)
        for i in range(50):
            learner.record(f"p{i}", f"response {i} long enough text here")
        model.learn.assert_not_called()
        assert learner.pending_count() == 50

    def test_batch_size_one_triggers_every_record(self, tmp_path: Path) -> None:
        learner, model = _make_learner(tmp_path, batch_size=1, min_response_length=1)
        for i in range(3):
            _record_and_join(learner, f"p{i}", "a")
        assert model.learn.call_count == 3
        assert learner.pending_count() == 0

    def test_join_on_no_thread_is_safe(self, tmp_path: Path) -> None:
        learner, _ = _make_learner(tmp_path, batch_size=100)
        # _join() before any training has been triggered must not raise
        learner._join()

    def test_lock_released_when_thread_start_fails(self, tmp_path: Path) -> None:
        # If Thread.start() raises (e.g. during interpreter shutdown), the
        # model lock must be released so subsequent record() calls can
        # still trigger training.
        learner, model = _make_learner(tmp_path, batch_size=2)
        learner.record("p1", "response long enough here")

        with patch("threading.Thread") as mock_thread_cls:
            bad_thread = MagicMock()
            bad_thread.start.side_effect = RuntimeError("cannot start thread")
            mock_thread_cls.return_value = bad_thread

            with pytest.raises(RuntimeError, match="cannot start thread"):
                learner.record("p2", "response long enough here")

        # Model lock must be free — a subsequent batch should be trainable.
        # The mock lock's acquire was called, so we verify it was released
        model._model_lock.release.assert_called()


# ── thread-safety tests ──────────────────────────────────────────────────────


class TestThreadSafety:
    def test_training_skipped_when_model_lock_held(self, tmp_path: Path) -> None:
        """Training should not start if model lock is already held (inference in progress)."""
        learner, model = _make_learner(tmp_path, batch_size=2)
        # Simulate the lock being held (e.g., by an ongoing inference)
        model._model_lock.acquire.return_value = False
        learner.record("p1", "response one long enough text")
        learner.record("p2", "response two long enough text")
        # Training should not be triggered because lock is held
        model.learn.assert_not_called()

    def test_training_proceeds_when_model_lock_free(self, tmp_path: Path) -> None:
        """Training should start when model lock is available."""
        learner, model = _make_learner(tmp_path, batch_size=2)
        # Lock is available
        model._model_lock.acquire.return_value = True
        learner.record("p1", "response one long enough text")
        learner.record("p2", "response two long enough text")
        # Training should be triggered - need to wait for thread
        learner._join()
        model.learn.assert_called_once()

    def test_concurrent_training_prevented(self, tmp_path: Path) -> None:
        """Only one training thread should run at a time."""
        learner, model = _make_learner(tmp_path, batch_size=2)
        # Simulate an active training thread
        learner._training_thread = MagicMock()
        learner._training_thread.is_alive.return_value = True

        learner.record("p1", "response one long enough text")
        learner.record("p2", "response two long enough text")
        # Training should not be triggered because thread is alive
        model.learn.assert_not_called()

    def test_lock_acquire_called_in_maybe_trigger_training(self, tmp_path: Path) -> None:
        """_maybe_trigger_training should check the model lock before spawning thread."""
        learner, model = _make_learner(tmp_path, batch_size=2)
        learner.record("p1", "response one long enough text")
        learner.record("p2", "response two long enough text")
        # The lock acquire should have been called
        model._model_lock.acquire.assert_called()
