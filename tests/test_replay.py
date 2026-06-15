"""Tests for ReplayBuffer — reservoir sampling, persistence, and learn() integration."""

from __future__ import annotations

import json
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# ── ReplayBuffer unit tests ───────────────────────────────────────────────────


class TestReplayBufferAdd:
    def test_add_fills_buffer_up_to_max_size(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("test/model", max_size=5, base_dir=tmp_path)
        buf.add([f"example {i}" for i in range(10)])
        assert len(buf) == 5

    def test_add_below_max_keeps_all(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("test/model", max_size=10, base_dir=tmp_path)
        buf.add(["a", "b", "c"])
        assert len(buf) == 3

    def test_total_seen_tracks_all_examples(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("test/model", max_size=3, base_dir=tmp_path)
        buf.add(["x", "y", "z", "w", "v"])
        assert buf.total_seen == 5

    def test_add_empty_list_is_noop(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("test/model", max_size=5, base_dir=tmp_path)
        buf.add([])
        assert len(buf) == 0
        assert buf.total_seen == 0


class TestReplayBufferSample:
    def test_sample_returns_correct_count(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("test/model", max_size=20, base_dir=tmp_path)
        buf.add([f"item {i}" for i in range(20)])
        assert len(buf.sample(5)) == 5

    def test_sample_capped_at_buffer_size(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("test/model", max_size=5, base_dir=tmp_path)
        buf.add(["a", "b", "c"])
        assert len(buf.sample(100)) == 3

    def test_sample_empty_buffer_returns_empty(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("test/model", max_size=10, base_dir=tmp_path)
        assert buf.sample(5) == []

    def test_sample_returns_strings(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("test/model", max_size=10, base_dir=tmp_path)
        buf.add(["hello", "world"])
        for item in buf.sample(2):
            assert isinstance(item, str)


class TestReplayBufferPersistence:
    def test_buffer_persists_across_instances(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf1 = ReplayBuffer("test/model", max_size=10, base_dir=tmp_path)
        buf1.add(["alpha", "beta", "gamma"])

        buf2 = ReplayBuffer("test/model", max_size=10, base_dir=tmp_path)
        assert len(buf2) == 3
        assert set(buf2.sample(3)) == {"alpha", "beta", "gamma"}

    def test_total_seen_persists(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf1 = ReplayBuffer("test/model", max_size=2, base_dir=tmp_path)
        buf1.add(["a", "b", "c", "d", "e"])

        buf2 = ReplayBuffer("test/model", max_size=2, base_dir=tmp_path)
        assert buf2.total_seen == 5

    def test_buffer_file_is_valid_jsonl(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("test/model", max_size=5, base_dir=tmp_path)
        buf.add(["line one", "line two"])

        lines = (tmp_path / "test--model" / "buffer.jsonl").read_text().splitlines()
        assert len(lines) == 3  # meta line + 2 text lines
        for line in lines:
            json.loads(line)  # must not raise

    def test_corrupt_buffer_file_resets_gracefully(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf_path = tmp_path / "test--model" / "buffer.jsonl"
        buf_path.parent.mkdir(parents=True)
        buf_path.write_text("not valid json\n")

        buf = ReplayBuffer("test/model", max_size=5, base_dir=tmp_path)
        assert len(buf) == 0
        assert buf.total_seen == 0


class TestReplayBufferDeduplication:
    def test_duplicate_texts_not_added(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("test/model", max_size=10, base_dir=tmp_path)
        buf.add(["alpha", "beta", "gamma"])
        buf.add(["alpha", "beta"])  # duplicates — should be skipped
        assert len(buf) == 3

    def test_total_seen_excludes_duplicates(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("test/model", max_size=10, base_dir=tmp_path)
        buf.add(["x", "y"])
        buf.add(["x", "z"])  # "x" is a duplicate
        assert buf.total_seen == 3  # only 3 unique texts

    def test_deduplication_persists_across_instances(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf1 = ReplayBuffer("test/model", max_size=10, base_dir=tmp_path)
        buf1.add(["hello", "world"])

        buf2 = ReplayBuffer("test/model", max_size=10, base_dir=tmp_path)
        buf2.add(["hello", "new"])  # "hello" already in persisted buffer
        assert len(buf2) == 3  # hello + world + new, not 4

    def test_duplicate_is_logged_at_debug(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("test/model", max_size=10, base_dir=tmp_path)
        buf.add(["dup"])
        with patch("pyrecall.replay.logger") as mock_logger:
            buf.add(["dup"])
            mock_logger.debug.assert_called_once()
            mock_logger.warning.assert_not_called()

    def test_clear_resets_deduplication(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("test/model", max_size=10, base_dir=tmp_path)
        buf.add(["foo", "bar"])
        buf.clear()
        buf.add(["foo", "bar"])  # after clear, these should be accepted again
        assert len(buf) == 2


class TestReplayBufferClear:
    def test_clear_empties_buffer(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("test/model", max_size=10, base_dir=tmp_path)
        buf.add(["x", "y"])
        buf.clear()
        assert len(buf) == 0
        assert buf.total_seen == 0

    def test_clear_persists(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("test/model", max_size=10, base_dir=tmp_path)
        buf.add(["x"])
        buf.clear()

        buf2 = ReplayBuffer("test/model", max_size=10, base_dir=tmp_path)
        assert len(buf2) == 0


class TestReplayBufferReservoirSampling:
    def test_reservoir_all_items_have_chance_to_be_retained(self, tmp_path: Path) -> None:
        """With max_size=5 and 50 items, each item should sometimes end up in the buffer."""
        from pyrecall.replay import ReplayBuffer

        random.seed(42)
        seen: set[str] = set()
        for _ in range(200):
            buf = ReplayBuffer("test/model", max_size=5, base_dir=tmp_path)
            buf.add([str(i) for i in range(50)])
            seen.update(buf.sample(5))
            buf.clear()

        # After 200 runs most of the 50 possible values should have appeared.
        assert len(seen) >= 40

    def test_max_size_property(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("test/model", max_size=42, base_dir=tmp_path)
        assert buf.max_size == 42


# ── learn() replay integration tests ─────────────────────────────────────────


def _make_mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.pad_token = None
    tok.eos_token = "<eos>"
    tok.eos_token_id = 0
    token_out = MagicMock()
    token_out.__getitem__ = lambda self, key: MagicMock(
        shape=torch.Size([1, 8]), to=lambda d: token_out
    )
    token_out.to = lambda d: token_out
    token_out.input_ids = torch.zeros(1, 8, dtype=torch.long)
    token_out.attention_mask = torch.ones(1, 8, dtype=torch.long)
    tok.return_value = token_out
    tok.decode.return_value = "ok"
    return tok


def _make_mock_base_model() -> MagicMock:
    base = MagicMock()
    base.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
    base.config = MagicMock()
    base.config.model_type = "gpt2"
    return base


def _make_mock_peft_model() -> MagicMock:
    peft = MagicMock()
    peft.parameters.return_value = [
        torch.nn.Parameter(torch.randn(10, 10)),
        torch.nn.Parameter(torch.randn(5, 5)),
    ]
    for p in peft.parameters():
        p.requires_grad = True
    hidden = torch.randn(1, 8, 32)
    outputs = MagicMock()
    outputs.hidden_states = [hidden] * 4
    peft.return_value = outputs
    peft.generate.return_value = torch.zeros(1, 10, dtype=torch.long)
    peft.eval.return_value = peft
    peft.train.return_value = peft
    peft.to.return_value = peft
    peft.save_pretrained = MagicMock()
    return peft


@pytest.fixture()
def patched_model_with_replay(tmp_path: Path):
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    mock_tok = _make_mock_tokenizer()
    mock_base = _make_mock_base_model()
    mock_peft = _make_mock_peft_model()

    with (
        patch("pyrecall.model.AutoTokenizer.from_pretrained", return_value=mock_tok),
        patch("pyrecall.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
        patch("pyrecall.model.get_peft_model", return_value=mock_peft),
        patch("pyrecall.model.compute_embeddings", return_value=torch.randn(32)),
        patch("pyrecall.model.cosine_similarity", return_value=0.75),
    ):
        from pyrecall.model import Model

        m = Model(
            "test/model",
            snapshot_dir=snap_dir,
            replay_buffer_size=50,
            replay_mix_ratio=0.5,
        )
        m.model = mock_peft
        yield m


class TestLearnReplayIntegration:
    def _run_learn(self, model, data_file: Path) -> MagicMock:
        from datasets import Dataset as HFDataset

        real_ds = HFDataset.from_dict(
            {"text": ["sample one", "sample two", "sample three", "sample four"]}
        )

        mock_trainer = MagicMock()
        with (
            patch("pyrecall.model.load_dataset", return_value=real_ds),
            patch("pyrecall.model.Trainer", return_value=mock_trainer),
            patch("pyrecall.model.TrainingArguments"),
            patch("pyrecall.model.DataCollatorForLanguageModeling"),
        ):
            # Patch tokenizer to return a simple tokenized dataset.
            model.tokenizer.side_effect = None
            model.tokenizer.return_value = {"input_ids": [1, 2], "attention_mask": [1, 1]}
            model.learn(str(data_file), epochs=1)
        return mock_trainer

    def test_buffer_populated_after_first_learn(
        self, patched_model_with_replay, tmp_path: Path
    ) -> None:
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(json.dumps({"text": "hi"}) + "\n")
        self._run_learn(patched_model_with_replay, data_file)
        assert len(patched_model_with_replay.replay_buffer) > 0

    def test_replay_disabled_when_buffer_size_zero(self, tmp_path: Path) -> None:
        snap_dir = tmp_path / "snapshots"
        snap_dir.mkdir()
        mock_tok = _make_mock_tokenizer()
        mock_base = _make_mock_base_model()
        mock_peft = _make_mock_peft_model()

        with (
            patch("pyrecall.model.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch("pyrecall.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
            patch("pyrecall.model.get_peft_model", return_value=mock_peft),
        ):
            from pyrecall.model import Model

            m = Model("test/model", snapshot_dir=snap_dir, replay_buffer_size=0)
            assert m.replay_buffer is None

    def test_replay_mixes_into_dataset_on_second_learn(
        self, patched_model_with_replay, tmp_path: Path
    ) -> None:
        from datasets import Dataset as HFDataset

        data_file = tmp_path / "train.jsonl"
        data_file.write_text(json.dumps({"text": "first run"}) + "\n")

        # Seed the buffer so mixing is triggered.
        patched_model_with_replay.replay_buffer.add([f"past example {i}" for i in range(10)])

        real_ds = HFDataset.from_dict({"text": ["a", "b", "c", "d"]})

        concatenate_called_with: list = []
        original_concat = __import__(
            "datasets", fromlist=["concatenate_datasets"]
        ).concatenate_datasets

        def spy_concat(datasets_list, **kwargs):
            concatenate_called_with.extend(datasets_list)
            return original_concat(datasets_list, **kwargs)

        mock_trainer = MagicMock()
        with (
            patch("pyrecall.model.load_dataset", return_value=real_ds),
            patch("pyrecall.model.Trainer", return_value=mock_trainer),
            patch("pyrecall.model.TrainingArguments"),
            patch("pyrecall.model.DataCollatorForLanguageModeling"),
            patch("pyrecall.model.concatenate_datasets", side_effect=spy_concat),
        ):
            patched_model_with_replay.tokenizer.side_effect = None
            patched_model_with_replay.tokenizer.return_value = {
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
            }
            patched_model_with_replay.learn(str(data_file), epochs=1)

        assert len(concatenate_called_with) >= 2


# ── Weighted replay tests ─────────────────────────────────────────────────────


class TestReplayBufferCategories:
    def test_add_stores_category_per_entry(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("test/model", max_size=10, base_dir=tmp_path)
        buf.add(["text a", "text b"], categories=["coding", "safety"])
        # sample returns text strings, so verify via persistence file
        lines = (tmp_path / "test--model" / "buffer.jsonl").read_text().splitlines()
        entries = [json.loads(line) for line in lines[1:] if line.strip()]
        cats = {e["text"]: e["category"] for e in entries}
        assert cats["text a"] == "coding"
        assert cats["text b"] == "safety"

    def test_add_without_categories_stores_none(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("test/model", max_size=10, base_dir=tmp_path)
        buf.add(["hello"])
        lines = (tmp_path / "test--model" / "buffer.jsonl").read_text().splitlines()
        entry = json.loads(lines[1])
        assert entry["category"] is None

    def test_backward_compat_load_old_format(self, tmp_path: Path) -> None:
        """Old buffer files without 'category' key load without error."""
        from pyrecall.replay import ReplayBuffer

        buf_path = tmp_path / "test--model" / "buffer.jsonl"
        buf_path.parent.mkdir(parents=True)
        buf_path.write_text(
            json.dumps({"total_seen": 2, "max_size": 10})
            + "\n"
            + json.dumps({"text": "old entry"})
            + "\n"
        )

        buf = ReplayBuffer("test/model", max_size=10, base_dir=tmp_path)
        assert len(buf) == 1
        assert buf.sample(1) == ["old entry"]

    def test_weighted_sample_up_weights_category(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        random.seed(0)
        buf = ReplayBuffer("test/model", max_size=200, base_dir=tmp_path)
        buf.add([f"coding {i}" for i in range(50)], categories=["coding"] * 50)
        buf.add([f"safety {i}" for i in range(50)], categories=["safety"] * 50)

        # Up-weight coding 4× vs safety 1×
        counts: dict[str, int] = {"coding": 0, "safety": 0}
        for _ in range(50):
            sample = buf.sample(20, weights={"coding": 4.0, "safety": 1.0})
            for text in sample:
                cat = text.split()[0]
                counts[cat] += 1

        # coding should appear more often than safety across all draws
        assert counts["coding"] > counts["safety"]

    def test_weighted_sample_falls_back_to_uniform_on_no_weights(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("test/model", max_size=10, base_dir=tmp_path)
        buf.add(["a", "b", "c"])
        result = buf.sample(3, weights=None)
        assert set(result) == {"a", "b", "c"}

    def test_learn_passes_categories_to_buffer(
        self, patched_model_with_replay, tmp_path: Path
    ) -> None:
        from unittest.mock import MagicMock, patch

        from datasets import Dataset as HFDataset

        data_file = tmp_path / "train.jsonl"
        data_file.write_text(json.dumps({"text": "train text", "category": "coding"}) + "\n")

        real_ds = HFDataset.from_dict({"text": ["train text"], "category": ["coding"]})
        mock_trainer = MagicMock()

        add_calls: list = []
        original_add = patched_model_with_replay.replay_buffer.add

        def spy_add(examples, categories=None):
            add_calls.append((examples, categories))
            return original_add(examples, categories=categories)

        patched_model_with_replay.replay_buffer.add = spy_add

        with (
            patch("pyrecall.model.load_dataset", return_value=real_ds),
            patch("pyrecall.model.Trainer", return_value=mock_trainer),
            patch("pyrecall.model.TrainingArguments"),
            patch("pyrecall.model.DataCollatorForLanguageModeling"),
        ):
            patched_model_with_replay.tokenizer.side_effect = None
            patched_model_with_replay.tokenizer.return_value = {
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
            }
            patched_model_with_replay.learn(str(data_file), epochs=1)

        assert add_calls, "add() was never called"
        _, categories = add_calls[0]
        assert categories is not None
        assert "coding" in categories

    def test_learn_replay_weights_from_report(
        self, patched_model_with_replay, tmp_path: Path
    ) -> None:
        from unittest.mock import MagicMock, patch

        from datasets import Dataset as HFDataset

        from pyrecall.detector import CategoryComparison, ForgettingReport

        data_file = tmp_path / "train.jsonl"
        data_file.write_text(json.dumps({"text": "hi"}) + "\n")

        patched_model_with_replay.replay_buffer.add(
            [f"coding {i}" for i in range(10)], categories=["coding"] * 10
        )

        report = ForgettingReport(
            snapshot_before="pre",
            snapshot_after="post",
            threshold=0.1,
            comparisons=[
                CategoryComparison(
                    category="coding", score_before=0.8, score_after=0.3, cohen_d=-1.2
                ),
            ],
        )

        real_ds = HFDataset.from_dict({"text": ["hi"]})
        mock_trainer = MagicMock()
        sample_calls: list = []
        original_sample = patched_model_with_replay.replay_buffer.sample

        def spy_sample(n, weights=None):
            sample_calls.append(weights)
            return original_sample(n, weights=weights)

        patched_model_with_replay.replay_buffer.sample = spy_sample

        with (
            patch("pyrecall.model.load_dataset", return_value=real_ds),
            patch("pyrecall.model.Trainer", return_value=mock_trainer),
            patch("pyrecall.model.TrainingArguments"),
            patch("pyrecall.model.DataCollatorForLanguageModeling"),
        ):
            patched_model_with_replay.tokenizer.side_effect = None
            patched_model_with_replay.tokenizer.return_value = {
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
            }
            patched_model_with_replay.learn(str(data_file), replay_weights=report)

        assert sample_calls, "sample() was never called"
        weights_used = sample_calls[0]
        assert weights_used is not None
        assert weights_used.get("coding", 1.0) > 1.0


class TestWeightedSampleEdgeCases:
    """_weighted_sample_without_replacement must always return exactly k items."""

    def test_all_zero_weights_returns_full_count(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("m", max_size=10, base_dir=tmp_path)
        buf.add(["a", "b", "c", "d", "e"], categories=["coding"] * 5)
        result = buf.sample(3, weights={"coding": 0.0})
        assert len(result) == 3

    def test_all_zero_weights_returns_strings(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("m", max_size=10, base_dir=tmp_path)
        buf.add(["x", "y", "z"], categories=["coding"] * 3)
        result = buf.sample(3, weights={"coding": 0.0})
        assert all(isinstance(s, str) for s in result)
        assert set(result) <= {"x", "y", "z"}

    def test_mixed_weights_some_zero_returns_full_count(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("m", max_size=10, base_dir=tmp_path)
        buf.add(["a", "b", "c"], categories=["coding", "safety", "coding"])
        result = buf.sample(3, weights={"coding": 0.0, "safety": 1.0})
        assert len(result) == 3

    def test_float_edge_case_inner_loop_always_selects(self, tmp_path: Path) -> None:
        from pyrecall.replay import _weighted_sample_without_replacement

        pool = [{"text": str(i), "category": None} for i in range(5)]
        weights = [0.1, 0.2, 0.3, 0.2, 0.2]
        for _ in range(50):
            result = _weighted_sample_without_replacement(list(pool), list(weights), k=5)
            assert len(result) == 5, f"expected 5 items, got {len(result)}: {result}"

    def test_sample_k_equals_buffer_size_returns_all(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("m", max_size=10, base_dir=tmp_path)
        texts = [f"text_{i}" for i in range(5)]
        buf.add(texts, categories=["coding"] * 5)
        result = buf.sample(5, weights={"coding": 0.0})
        assert len(result) == 5


class TestCorruptBufferLoad:
    def test_corrupt_entry_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("m", max_size=10, base_dir=tmp_path)
        buf.add(["good entry"])
        # Inject a corrupt line directly into the file
        buf._path.write_text(buf._path.read_text().rstrip("\n") + "\nnot json at all\n")
        buf2 = ReplayBuffer("m", max_size=10, base_dir=tmp_path)
        assert len(buf2) == 1
        assert buf2.sample(1) == ["good entry"]

    def test_entry_missing_text_key_is_skipped(self, tmp_path: Path) -> None:
        import json

        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("m", max_size=10, base_dir=tmp_path)
        buf.add(["good"])
        buf._path.write_text(
            buf._path.read_text().rstrip("\n") + "\n" + json.dumps({"category": "coding"}) + "\n"
        )
        buf2 = ReplayBuffer("m", max_size=10, base_dir=tmp_path)
        assert len(buf2) == 1

    def test_all_zero_weights_empty_pool_does_not_crash(self, tmp_path: Path) -> None:
        from pyrecall.replay import _weighted_sample_without_replacement

        result = _weighted_sample_without_replacement([], [], k=3)
        assert result == []


class TestSeenHashesPersisted:
    def test_evicted_entry_not_readded_after_restart(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("m", max_size=2, base_dir=tmp_path)
        buf.add(["a", "b", "c"])  # c evicts one; total_seen=3, buffer has 2
        total_before = buf.total_seen

        buf2 = ReplayBuffer("m", max_size=2, base_dir=tmp_path)
        buf2.add(["a", "b", "c"])  # all three should be seen as duplicates
        assert buf2.total_seen == total_before  # no inflation

    def test_seen_hashes_survive_restart(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("m", max_size=10, base_dir=tmp_path)
        buf.add(["hello", "world"])

        buf2 = ReplayBuffer("m", max_size=10, base_dir=tmp_path)
        duplicates_before = buf2.total_seen
        buf2.add(["hello", "world"])
        assert buf2.total_seen == duplicates_before  # no re-count


class TestReplayBufferTrimOnLoad:
    """#132: trimming on load must use random.sample, not tail-slice."""

    def test_trim_preserves_random_distribution(self, tmp_path: Path) -> None:
        from pyrecall.replay import ReplayBuffer

        buf = ReplayBuffer("m", max_size=100, base_dir=tmp_path)
        buf.add([f"example {i}" for i in range(100)])

        # Reload with smaller max_size.
        buf2 = ReplayBuffer("m", max_size=10, base_dir=tmp_path)
        assert len(buf2) == 10
        texts = {e["text"] for e in buf2._buffer}
        # Entries should not all be from the first 10 (which would happen with slice).
        # With 100 items trimmed to 10, chance all 10 are the first 10 is astronomically small.
        first_ten = {f"example {i}" for i in range(10)}
        assert texts != first_ten
