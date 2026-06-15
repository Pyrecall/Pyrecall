"""Tests for SkillScore, SkillSnapshot save/load, and Encryptor."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from pyrecall.snapshot import SkillScore, SkillSnapshot

# ── helpers ────────────────────────────────────────────────────────────────────

_DT = datetime(2024, 6, 1, 12, 0, 0)


def _make_snapshot(
    name: str = "v1",
    model_name: str = "test/model",
    n_scores: int = 2,
) -> SkillSnapshot:
    scores = [
        SkillScore(
            category="reasoning" if i == 0 else "coding",
            prompt=f"prompt {i}",
            response=f"response {i}",
            score=0.8 + i * 0.05,
        )
        for i in range(n_scores)
    ]
    return SkillSnapshot(name=name, model_name=model_name, created_at=_DT, scores=scores)


# ── SkillScore ─────────────────────────────────────────────────────────────────


class TestSkillScore:
    def test_to_dict_round_trips(self) -> None:
        s = SkillScore(category="reasoning", prompt="p", response="r", score=0.75)
        assert SkillScore.from_dict(s.to_dict()) == s

    def test_to_dict_contains_all_keys(self) -> None:
        s = SkillScore(category="coding", prompt="p", response="r", score=0.5)
        d = s.to_dict()
        assert set(d) == {"category", "prompt", "response", "score", "scoring_method"}

    def test_from_dict_coerces_score_to_float(self) -> None:
        s = SkillScore.from_dict({"category": "c", "prompt": "p", "response": "r", "score": "0.9"})
        assert isinstance(s.score, float)
        assert s.score == 0.9


# ── SkillSnapshot aggregation ──────────────────────────────────────────────────


class TestSkillSnapshotConstruction:
    def test_positional_scores_arg_raises_type_error(self) -> None:
        score = SkillScore(category="safety", prompt="p", response="c", score=0.5)
        with pytest.raises(TypeError, match="created_at must be a datetime"):
            SkillSnapshot("v1", "llama", [score])

    def test_keyword_args_construct_correctly(self) -> None:
        score = SkillScore(category="safety", prompt="p", response="c", score=0.5)
        snap = SkillSnapshot(name="v1", model_name="llama", scores=[score])
        assert snap.scores == [score]
        assert isinstance(snap.created_at, datetime)


class TestSkillSnapshotAggregation:
    def test_overall_score_is_mean(self) -> None:
        snap = _make_snapshot()
        expected = (0.80 + 0.85) / 2
        assert abs(snap.overall_score() - expected) < 1e-9

    def test_overall_score_empty_is_zero(self) -> None:
        snap = SkillSnapshot(name="empty", model_name="m")
        assert snap.overall_score() == 0.0

    def test_category_scores_averages_per_category(self) -> None:
        scores = [
            SkillScore("cat", "p1", "r1", 0.6),
            SkillScore("cat", "p2", "r2", 0.8),
        ]
        snap = SkillSnapshot(name="s", model_name="m", scores=scores)
        assert abs(snap.category_scores()["cat"] - 0.7) < 1e-9

    def test_category_scores_multiple_categories(self) -> None:
        snap = _make_snapshot(n_scores=2)
        cats = snap.category_scores()
        assert set(cats) == {"reasoning", "coding"}


# ── SkillSnapshot save/load (plain) ───────────────────────────────────────────


class TestSkillSnapshotPersistence:
    def test_save_creates_snapshot_json(self, tmp_path: Path) -> None:
        snap = _make_snapshot()
        snap.save(tmp_path / "v1")
        assert (tmp_path / "v1" / "snapshot.json").exists()

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        snap = _make_snapshot()
        snap.save(tmp_path / "nested" / "v1")
        assert (tmp_path / "nested" / "v1").is_dir()

    def test_save_json_is_valid(self, tmp_path: Path) -> None:
        snap = _make_snapshot()
        snap.save(tmp_path / "v1")
        data = json.loads((tmp_path / "v1" / "snapshot.json").read_text())
        assert data["name"] == "v1"
        assert data["model_name"] == "test/model"
        assert data["encrypted"] is False

    def test_save_serialises_scores(self, tmp_path: Path) -> None:
        snap = _make_snapshot(n_scores=3)
        snap.save(tmp_path / "v1")
        data = json.loads((tmp_path / "v1" / "snapshot.json").read_text())
        assert len(data["scores"]) == 3

    def test_load_round_trips_name_and_model(self, tmp_path: Path) -> None:
        snap = _make_snapshot()
        snap.save(tmp_path / "v1")
        loaded = SkillSnapshot.load(tmp_path / "v1")
        assert loaded.name == "v1"
        assert loaded.model_name == "test/model"

    def test_load_round_trips_scores(self, tmp_path: Path) -> None:
        snap = _make_snapshot(n_scores=2)
        snap.save(tmp_path / "v1")
        loaded = SkillSnapshot.load(tmp_path / "v1")
        assert len(loaded.scores) == 2
        assert loaded.scores[0].category == "reasoning"

    def test_load_round_trips_created_at(self, tmp_path: Path) -> None:
        snap = _make_snapshot()
        snap.save(tmp_path / "v1")
        loaded = SkillSnapshot.load(tmp_path / "v1")
        assert loaded.created_at == _DT

    def test_load_round_trips_adapter_path(self, tmp_path: Path) -> None:
        adapter = tmp_path / "adapter"
        snap = _make_snapshot()
        snap.adapter_path = adapter
        snap.save(tmp_path / "v1")
        loaded = SkillSnapshot.load(tmp_path / "v1")
        assert loaded.adapter_path == adapter

    def test_load_adapter_path_none_when_absent(self, tmp_path: Path) -> None:
        snap = _make_snapshot()
        snap.save(tmp_path / "v1")
        loaded = SkillSnapshot.load(tmp_path / "v1")
        assert loaded.adapter_path is None

    def test_load_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="snapshot.json"):
            SkillSnapshot.load(tmp_path / "nonexistent")

    def test_load_raises_on_corrupt_json(self, tmp_path: Path) -> None:
        d = tmp_path / "bad"
        d.mkdir()
        (d / "snapshot.json").write_text("not json {{{")
        with pytest.raises(ValueError, match="corrupted"):
            SkillSnapshot.load(d)

    def test_encrypted_field_defaults_false(self) -> None:
        snap = _make_snapshot()
        assert snap.encrypted is False

    def test_save_load_preserves_score_values(self, tmp_path: Path) -> None:
        snap = _make_snapshot(n_scores=2)
        snap.save(tmp_path / "v1")
        loaded = SkillSnapshot.load(tmp_path / "v1")
        original_scores = {s.prompt: s.score for s in snap.scores}
        loaded_scores = {s.prompt: s.score for s in loaded.scores}
        assert original_scores == loaded_scores


# ── SkillSnapshot save/load (privacy=True) ────────────────────────────────────


class TestSkillSnapshotPrivacy:
    def _requires_cryptography(self) -> None:
        try:
            import cryptography  # noqa: F401
        except ImportError:
            pytest.skip("cryptography not installed")

    def test_privacy_save_creates_file(self, tmp_path: Path) -> None:
        self._requires_cryptography()
        snap = _make_snapshot()
        snap.save(tmp_path / "v1", privacy=True)
        assert (tmp_path / "v1" / "snapshot.json").exists()

    def test_privacy_save_sets_encrypted_true(self, tmp_path: Path) -> None:
        self._requires_cryptography()
        snap = _make_snapshot()
        snap.save(tmp_path / "v1", privacy=True)
        data = json.loads((tmp_path / "v1" / "snapshot.json").read_text())
        assert data["encrypted"] is True

    def test_privacy_save_stores_key(self, tmp_path: Path) -> None:
        self._requires_cryptography()
        snap = _make_snapshot()
        snap.save(tmp_path / "v1", privacy=True)
        data = json.loads((tmp_path / "v1" / "snapshot.json").read_text())
        assert "key" in data
        assert len(data["key"]) > 0

    def test_privacy_save_encrypts_name(self, tmp_path: Path) -> None:
        self._requires_cryptography()
        snap = _make_snapshot(name="secret-snap")
        snap.save(tmp_path / "v1", privacy=True)
        data = json.loads((tmp_path / "v1" / "snapshot.json").read_text())
        assert data["name"] != "secret-snap"

    def test_privacy_round_trip_name(self, tmp_path: Path) -> None:
        self._requires_cryptography()
        snap = _make_snapshot(name="secret-snap")
        snap.save(tmp_path / "v1", privacy=True)
        loaded = SkillSnapshot.load(tmp_path / "v1", privacy=True)
        assert loaded.name == "secret-snap"

    def test_privacy_round_trip_model_name(self, tmp_path: Path) -> None:
        self._requires_cryptography()
        snap = _make_snapshot(model_name="org/private-model")
        snap.save(tmp_path / "v1", privacy=True)
        loaded = SkillSnapshot.load(tmp_path / "v1", privacy=True)
        assert loaded.model_name == "org/private-model"

    def test_privacy_round_trip_scores(self, tmp_path: Path) -> None:
        self._requires_cryptography()
        snap = _make_snapshot(n_scores=3)
        snap.save(tmp_path / "v1", privacy=True)
        loaded = SkillSnapshot.load(tmp_path / "v1", privacy=True)
        assert len(loaded.scores) == 3
        assert loaded.scores[0].score == pytest.approx(snap.scores[0].score)

    def test_privacy_round_trip_created_at(self, tmp_path: Path) -> None:
        self._requires_cryptography()
        snap = _make_snapshot()
        snap.save(tmp_path / "v1", privacy=True)
        loaded = SkillSnapshot.load(tmp_path / "v1", privacy=True)
        assert loaded.created_at == _DT

    def test_privacy_round_trip_sets_encrypted_flag(self, tmp_path: Path) -> None:
        self._requires_cryptography()
        snap = _make_snapshot()
        snap.save(tmp_path / "v1", privacy=True)
        loaded = SkillSnapshot.load(tmp_path / "v1", privacy=True)
        assert loaded.encrypted is True

    def test_privacy_save_without_cryptography_raises(self, tmp_path: Path) -> None:
        with patch.dict("sys.modules", {"cryptography": None, "cryptography.fernet": None}):
            snap = _make_snapshot()
            with pytest.raises(ImportError, match="privacy"):
                snap.save(tmp_path / "v1", privacy=True)


# ── Encryptor ──────────────────────────────────────────────────────────────────


class TestEncryptor:
    def _requires_cryptography(self) -> None:
        try:
            import cryptography  # noqa: F401
        except ImportError:
            pytest.skip("cryptography not installed")

    def test_encrypt_decrypt_round_trips(self) -> None:
        self._requires_cryptography()
        from pyrecall.encrypt import Encryptor

        enc = Encryptor()
        original = "hello world"
        assert enc.decrypt(enc.encrypt(original)) == original

    def test_encrypt_produces_different_bytes(self) -> None:
        self._requires_cryptography()
        from pyrecall.encrypt import Encryptor

        enc = Encryptor()
        assert enc.encrypt("secret") != "secret"

    def test_same_key_can_decrypt(self) -> None:
        self._requires_cryptography()
        from pyrecall.encrypt import Encryptor

        enc1 = Encryptor()
        ciphertext = enc1.encrypt("data")
        enc2 = Encryptor(key=enc1.key)
        assert enc2.decrypt(ciphertext) == "data"

    def test_key_is_bytes(self) -> None:
        self._requires_cryptography()
        from pyrecall.encrypt import Encryptor

        enc = Encryptor()
        assert isinstance(enc.key, bytes)

    def test_missing_cryptography_raises_import_error(self) -> None:
        with patch.dict("sys.modules", {"cryptography": None, "cryptography.fernet": None}):
            import importlib

            import pyrecall.encrypt as enc_mod

            importlib.reload(enc_mod)
            with pytest.raises(ImportError, match="privacy"):
                enc_mod.Encryptor()


class TestOverallScoreCategoryBalanced:
    def test_overall_score_is_category_balanced(self) -> None:
        scores = [
            SkillScore(category="math", prompt=f"p{i}", response="r", score=0.9) for i in range(10)
        ] + [
            SkillScore(category="coding", prompt=f"q{i}", response="r", score=0.1) for i in range(2)
        ]
        snap = SkillSnapshot(name="s", model_name="m", scores=scores)
        # category-balanced: (0.9 + 0.1) / 2 = 0.5, not prompt-weighted ~0.85
        assert abs(snap.overall_score() - 0.5) < 1e-9

    def test_overall_score_equal_categories_unchanged(self) -> None:
        scores = [
            SkillScore(category="a", prompt="p1", response="r", score=0.8),
            SkillScore(category="b", prompt="p2", response="r", score=0.6),
        ]
        snap = SkillSnapshot(name="s", model_name="m", scores=scores)
        assert abs(snap.overall_score() - 0.7) < 1e-9

    def test_overall_score_empty_returns_zero(self) -> None:
        snap = SkillSnapshot(name="s", model_name="m")
        assert snap.overall_score() == 0.0

    def test_overall_score_single_category(self) -> None:
        scores = [
            SkillScore(category="coding", prompt=f"p{i}", response="r", score=0.8) for i in range(5)
        ]
        snap = SkillSnapshot(name="s", model_name="m", scores=scores)
        assert abs(snap.overall_score() - 0.8) < 1e-9
