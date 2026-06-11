"""
End-to-end smoke test — full pyrecall lifecycle on gpt2 (CPU).

Downloads gpt2 (~500 MB) on first run; cached afterwards.
Runs in about 3–5 minutes on a modern laptop CPU.

Run explicitly with:
    pytest tests/test_smoke_gpt2.py -v -s
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture()
def training_jsonl(tmp_path: Path) -> Path:
    """Write a tiny JSONL training file with 5 examples."""
    rows = [
        {"text": "### Human: What is 2 + 2?\n\n### Assistant: 4"},
        {"text": "### Human: Name the capital of France.\n\n### Assistant: Paris"},
        {"text": "### Human: What colour is the sky?\n\n### Assistant: Blue"},
        {"text": "### Human: How many days are in a week?\n\n### Assistant: 7"},
        {"text": "### Human: What is the boiling point of water in Celsius?\n\n### Assistant: 100 degrees Celsius"},
    ]
    p = tmp_path / "train.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_full_lifecycle(tmp_path: Path, training_jsonl: Path) -> None:
    """Load gpt2, snapshot, learn, check, rollback — all must complete without error."""
    from pyrecall.model import Model

    # ── 1. load ────────────────────────────────────────────────────────────────
    model = Model(
        "gpt2",
        strategy="lora",
        lora_r=4,           # small rank for speed
        lora_alpha=8,
        batch_size=1,
        max_length=128,
        snapshot_dir=tmp_path / "snapshots",
    )

    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    assert trainable > 0, "No trainable parameters — LoRA not applied"

    # ── 2. snapshot before ────────────────────────────────────────────────────
    before = model.snapshot("before")

    assert before.name == "before"
    assert len(before.scores) == 48, "Expected 48 benchmark scores"
    assert all(0.0 <= s.score <= 1.0 for s in before.scores), "Scores must be in [0, 1]"
    assert before.adapter_path is not None and before.adapter_path.exists(), (
        "Adapter weights not saved"
    )

    overall_before = before.overall_score()
    assert 0.0 <= overall_before <= 1.0

    # ── 3. learn ──────────────────────────────────────────────────────────────
    model.learn(str(training_jsonl), epochs=1)

    # ── 4. check ──────────────────────────────────────────────────────────────
    report = model.check()

    assert hasattr(report, "is_healthy"), "ForgettingReport missing is_healthy"
    assert hasattr(report, "degraded_skills")

    # Scores after training must also be valid
    after_snap = model.rollback_manager.load_snapshot("before__after")
    assert all(0.0 <= s.score <= 1.0 for s in after_snap.scores), (
        "Post-training scores out of [0, 1]"
    )

    # ── 5. rollback ───────────────────────────────────────────────────────────
    model.rollback(to="before")

    # Model must still generate text after rollback
    response = model.generate("What is 2 + 2?", max_new_tokens=20)
    assert isinstance(response, str), "generate() must return a string after rollback"


def test_snapshot_persisted_to_disk(tmp_path: Path) -> None:
    """Snapshot JSON and adapter weights must exist on disk after snapshot()."""
    from pyrecall.model import Model

    snap_dir = tmp_path / "snapshots"
    model = Model(
        "gpt2",
        lora_r=4,
        lora_alpha=8,
        batch_size=1,
        max_length=64,
        snapshot_dir=snap_dir,
    )
    model.snapshot("disk_check")

    snap_json = snap_dir / "disk_check" / "snapshot.json"
    adapter_dir = snap_dir / "disk_check" / "adapter"

    assert snap_json.exists(), "snapshot.json not written"
    data = json.loads(snap_json.read_text())
    assert data["name"] == "disk_check"
    assert data["model_name"] == "gpt2"
    assert isinstance(data["scores"], list) and len(data["scores"]) == 48

    assert adapter_dir.exists(), "adapter/ directory not saved"
    adapter_files = list(adapter_dir.iterdir())
    assert len(adapter_files) > 0, "adapter/ directory is empty"


def test_learn_rejects_missing_file(tmp_path: Path) -> None:
    """learn() must raise PyrecallError for a non-existent data file."""
    from pyrecall.model import Model, PyrecallError

    model = Model("gpt2", lora_r=4, lora_alpha=8, batch_size=1, max_length=64,
                  snapshot_dir=tmp_path / "snapshots")

    with pytest.raises(PyrecallError, match="not found"):
        model.learn(str(tmp_path / "nonexistent.jsonl"))


def test_generate_returns_string(tmp_path: Path) -> None:
    """generate() must return a non-empty string on a simple prompt."""
    from pyrecall.model import Model

    model = Model("gpt2", lora_r=4, lora_alpha=8, batch_size=1, max_length=64,
                  snapshot_dir=tmp_path / "snapshots")

    response = model.generate("The capital of France is", max_new_tokens=10)
    assert isinstance(response, str)
    assert len(response) > 0


def test_rollback_restores_adapter_weights(tmp_path: Path, training_jsonl: Path) -> None:
    """
    After rolling back to a pre-training snapshot, the model's LoRA weights must
    match those saved in the snapshot adapter directory, not the post-training weights.
    """
    from pyrecall.model import Model

    model = Model(
        "gpt2",
        lora_r=4,
        lora_alpha=8,
        batch_size=1,
        max_length=64,
        snapshot_dir=tmp_path / "snapshots",
    )

    # Snapshot before training and record a weight fingerprint
    model.snapshot("pre")
    pre_weight = next(p.clone().detach() for p in model.model.parameters() if p.requires_grad)

    # Train so weights change
    model.learn(str(training_jsonl), epochs=1)
    post_weight = next(p.clone().detach() for p in model.model.parameters() if p.requires_grad)

    # Weights must have changed after training
    assert not torch.allclose(pre_weight, post_weight, atol=1e-6), (
        "Weights didn't change after training — the smoke test is invalid"
    )

    # Rollback and verify weights match the pre-training state
    model.rollback(to="pre")
    rolled_back_weight = next(p.clone().detach() for p in model.model.parameters() if p.requires_grad)

    assert torch.allclose(pre_weight.cpu(), rolled_back_weight.cpu(), atol=1e-5), (
        "Rolled-back weights don't match the pre-training snapshot"
    )

    # Model must still generate text after rollback
    response = model.generate("hello", max_new_tokens=5)
    assert isinstance(response, str) and len(response) > 0
