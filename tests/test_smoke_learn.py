"""
End-to-end smoke tests for Model.learn() on gpt2 (CPU).

Covers the replay buffer path that test_smoke_gpt2.py doesn't exercise:
buffer population, replay mixing on a second run, and CSV/resume paths.

Run explicitly with:
    pytest tests/test_smoke_learn.py -v -s
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture()
def model(tmp_path: Path):
    from pyrecall.model import Model

    return Model(
        "gpt2",
        strategy="lora",
        lora_r=4,
        lora_alpha=8,
        batch_size=1,
        max_length=64,
        snapshot_dir=tmp_path / "snapshots",
        replay_buffer_size=20,
        replay_mix_ratio=0.5,
    )


@pytest.fixture()
def jsonl_file(tmp_path: Path) -> Path:
    rows = [
        {"text": "The sky is blue."},
        {"text": "Water boils at 100 degrees."},
        {"text": "Paris is the capital of France."},
        {"text": "2 + 2 equals 4."},
        {"text": "The Earth orbits the Sun."},
    ]
    p = tmp_path / "train.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_learn_completes_without_error(model, jsonl_file: Path) -> None:
    """learn() must run to completion with no exception."""
    model.learn(str(jsonl_file), epochs=1)


def test_replay_buffer_populated_after_learn(model, jsonl_file: Path) -> None:
    """Buffer must contain examples after a learn() call."""
    assert model.replay_buffer is not None
    assert len(model.replay_buffer) == 0

    model.learn(str(jsonl_file), epochs=1)

    assert len(model.replay_buffer) > 0


def test_replay_buffer_does_not_exceed_max_size(tmp_path: Path, jsonl_file: Path) -> None:
    """Buffer must never exceed its configured max_size."""
    from pyrecall.model import Model

    model = Model(
        "gpt2",
        lora_r=4,
        lora_alpha=8,
        batch_size=1,
        max_length=64,
        snapshot_dir=tmp_path / "snapshots",
        replay_buffer_size=3,
    )
    model.learn(str(jsonl_file), epochs=1)
    assert len(model.replay_buffer) <= 3


def test_second_learn_uses_replay_examples(model, jsonl_file: Path, tmp_path: Path) -> None:
    """On the second learn() call the replay buffer must be non-empty going in."""
    model.learn(str(jsonl_file), epochs=1)
    size_after_first = len(model.replay_buffer)
    assert size_after_first > 0

    second_file = tmp_path / "train2.jsonl"
    second_file.write_text(json.dumps({"text": "The Moon orbits the Earth."}) + "\n")

    model.learn(str(second_file), epochs=1)
    assert len(model.replay_buffer) > 0


def test_learn_disabled_replay_buffer(tmp_path: Path, jsonl_file: Path) -> None:
    """replay_buffer_size=0 means no buffer is created and learn still works."""
    from pyrecall.model import Model

    model = Model(
        "gpt2",
        lora_r=4,
        lora_alpha=8,
        batch_size=1,
        max_length=64,
        snapshot_dir=tmp_path / "snapshots",
        replay_buffer_size=0,
    )
    assert model.replay_buffer is None
    model.learn(str(jsonl_file), epochs=1)


def test_learn_resume_from_checkpoint(model, jsonl_file: Path, tmp_path: Path) -> None:
    """resume=True must not raise even when no checkpoint exists (starts fresh)."""
    model.learn(str(jsonl_file), epochs=1, resume=True)


def test_learn_csv_format(model, tmp_path: Path) -> None:
    """learn() must accept a CSV file with a 'text' column."""
    csv_file = tmp_path / "train.csv"
    csv_file.write_text("text\nHello world.\nGoodbye world.\nFoo bar.\n")
    model.learn(str(csv_file), epochs=1)


def test_learn_unsupported_format_raises(model, tmp_path: Path) -> None:
    """learn() must raise PyrecallError for an unsupported file extension."""
    from pyrecall.model import PyrecallError

    bad_file = tmp_path / "data.txt"
    bad_file.write_text("some text\n")
    with pytest.raises(PyrecallError, match="Unsupported file format"):
        model.learn(str(bad_file), epochs=1)
