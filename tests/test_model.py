"""Tests for the Model class — HuggingFace/PEFT calls are mocked for speed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch


# ── fixtures ──────────────────────────────────────────────────────────────────


def _make_mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.pad_token = None
    tok.eos_token = "<eos>"
    tok.eos_token_id = 0
    # Simulate tokenizer call returning tensors
    token_out = MagicMock()
    token_out.__getitem__ = lambda self, key: MagicMock(
        shape=torch.Size([1, 8]), to=lambda d: token_out
    )
    token_out.to = lambda d: token_out
    token_out.input_ids = torch.zeros(1, 8, dtype=torch.long)
    token_out.attention_mask = torch.ones(1, 8, dtype=torch.long)
    tok.return_value = token_out
    tok.decode.return_value = "Paris is the capital of France."
    return tok


def _make_mock_base_model() -> MagicMock:
    base = MagicMock()
    base.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
    # Make base_model attribute accessible for PEFT wrapping
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
def tmp_snapshot_dir(tmp_path: Path) -> Path:
    d = tmp_path / "snapshots"
    d.mkdir()
    return d


@pytest.fixture()
def patched_model(tmp_snapshot_dir: Path):
    """Model instance with all HuggingFace and PEFT calls mocked."""
    mock_tokenizer = _make_mock_tokenizer()
    mock_base = _make_mock_base_model()
    mock_peft = _make_mock_peft_model()

    with (
        patch("keel.model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("keel.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
        patch("keel.model.get_peft_model", return_value=mock_peft),
        patch("keel.model.compute_embeddings", return_value=torch.randn(32)),
        patch("keel.model.cosine_similarity", return_value=0.75),
    ):
        from keel.model import Model

        m = Model("test/model", snapshot_dir=tmp_snapshot_dir)
        m.model = mock_peft
        yield m


# ── tests ──────────────────────────────────────────────────────────────────────


class TestModelInit:
    def test_invalid_strategy_raises(self, tmp_snapshot_dir: Path) -> None:
        mock_tok = _make_mock_tokenizer()
        mock_base = _make_mock_base_model()
        mock_peft = _make_mock_peft_model()

        with (
            patch("keel.model.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch("keel.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
            patch("keel.model.get_peft_model", return_value=mock_peft),
        ):
            from keel.model import KeelError, Model

            with pytest.raises(KeelError, match="strategy"):
                Model("test/model", strategy="full", snapshot_dir=tmp_snapshot_dir)

    def test_pad_token_set_when_missing(self, patched_model) -> None:
        # Tokenizer pad_token was None; should have been set to eos_token.
        assert patched_model.tokenizer.pad_token == "<eos>"


class TestModelGenerate:
    def test_returns_decoded_string(self, patched_model) -> None:
        result = patched_model.generate("What is 2+2?")
        assert isinstance(result, str)

    def test_generate_calls_tokenizer(self, patched_model) -> None:
        patched_model.generate("Hello")
        patched_model.tokenizer.assert_called()


class TestModelSnapshot:
    def test_snapshot_saves_json(self, patched_model, tmp_snapshot_dir: Path) -> None:
        snap = patched_model.snapshot(name="test_snap")
        snap_file = tmp_snapshot_dir / "test_snap" / "snapshot.json"
        assert snap_file.exists(), "snapshot.json must be written to disk"

    def test_snapshot_sets_baseline(self, patched_model) -> None:
        patched_model.snapshot(name="baseline")
        assert patched_model._baseline_snapshot_name == "baseline"

    def test_snapshot_returns_skill_snapshot(self, patched_model) -> None:
        from keel.snapshot import SkillSnapshot

        snap = patched_model.snapshot(name="v1")
        assert isinstance(snap, SkillSnapshot)

    def test_snapshot_has_correct_score_count(self, patched_model) -> None:
        from keel.benchmarks.default import DEFAULT_BENCHMARKS

        snap = patched_model.snapshot(name="count_test")
        assert len(snap.scores) == len(DEFAULT_BENCHMARKS)

    def test_snapshot_scores_normalised(self, patched_model) -> None:
        snap = patched_model.snapshot(name="norm_test")
        for score_item in snap.scores:
            assert 0.0 <= score_item.score <= 1.0


class TestModelCheck:
    def test_check_raises_without_baseline(self, tmp_snapshot_dir: Path) -> None:
        mock_tok = _make_mock_tokenizer()
        mock_base = _make_mock_base_model()
        mock_peft = _make_mock_peft_model()

        with (
            patch("keel.model.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch("keel.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
            patch("keel.model.get_peft_model", return_value=mock_peft),
            patch("keel.model.compute_embeddings", return_value=torch.randn(32)),
            patch("keel.model.cosine_similarity", return_value=0.75),
        ):
            from keel.model import KeelError, Model

            m = Model("test/model", snapshot_dir=tmp_snapshot_dir)
            with pytest.raises(KeelError, match="snapshot"):
                m.check()

    def test_check_returns_report(self, patched_model) -> None:
        from keel.detector import ForgettingReport

        patched_model.snapshot(name="pre")
        report = patched_model.check()
        assert isinstance(report, ForgettingReport)

    def test_check_report_has_comparisons(self, patched_model) -> None:
        patched_model.snapshot(name="pre2")
        report = patched_model.check()
        assert len(report.comparisons) > 0


class TestModelLearn:
    def test_learn_raises_for_missing_file(self, patched_model) -> None:
        from keel.model import KeelError

        with pytest.raises(KeelError, match="not found"):
            patched_model.learn("/nonexistent/data.jsonl")

    def test_learn_runs_with_valid_jsonl(
        self, patched_model, tmp_path: Path
    ) -> None:
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(
            json.dumps({"text": "### Human: Hi\n\n### Assistant: Hello!"}) + "\n"
        )

        mock_trainer = MagicMock()
        mock_trainer.train = MagicMock()

        with (
            patch("keel.model.load_dataset") as mock_ds,
            patch("keel.model.Trainer", return_value=mock_trainer),
            patch("keel.model.TrainingArguments"),
            patch("keel.model.DataCollatorForLanguageModeling"),
        ):
            mock_dataset = MagicMock()
            mock_dataset.column_names = ["text"]
            mock_dataset.map.return_value = mock_dataset
            mock_ds.return_value = mock_dataset

            patched_model.learn(str(data_file), epochs=1)
            mock_trainer.train.assert_called_once()


class TestLearnDataFormats:
    """learn() must route .csv and .parquet to the right load_dataset format."""

    def _run_learn(self, patched_model, data_file: Path) -> MagicMock:
        mock_trainer = MagicMock()
        with (
            patch("keel.model.load_dataset") as mock_ds,
            patch("keel.model.Trainer", return_value=mock_trainer),
            patch("keel.model.TrainingArguments"),
            patch("keel.model.DataCollatorForLanguageModeling"),
        ):
            mock_dataset = MagicMock()
            mock_dataset.column_names = ["text"]
            mock_dataset.map.return_value = mock_dataset
            mock_ds.return_value = mock_dataset

            patched_model.learn(str(data_file), epochs=1)
            return mock_ds

    def test_jsonl_uses_json_format(self, patched_model, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"text": "hi"}) + "\n")
        mock_ds = self._run_learn(patched_model, f)
        mock_ds.assert_called_once()
        assert mock_ds.call_args[0][0] == "json"

    def test_csv_uses_csv_format(self, patched_model, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("text\nhello world\n")
        mock_ds = self._run_learn(patched_model, f)
        mock_ds.assert_called_once()
        assert mock_ds.call_args[0][0] == "csv"

    def test_parquet_uses_parquet_format(self, patched_model, tmp_path: Path) -> None:
        f = tmp_path / "data.parquet"
        f.write_bytes(b"fake-parquet-bytes")
        mock_ds = self._run_learn(patched_model, f)
        mock_ds.assert_called_once()
        assert mock_ds.call_args[0][0] == "parquet"

    def test_unsupported_format_raises(self, patched_model, tmp_path: Path) -> None:
        from keel.model import KeelError

        f = tmp_path / "data.txt"
        f.write_text("hello\n")
        with pytest.raises(KeelError, match="Unsupported file format"):
            patched_model.learn(str(f), epochs=1)


class TestQLoRA:
    def test_qlora_strategy_accepted(self, tmp_snapshot_dir: Path) -> None:
        mock_tok = _make_mock_tokenizer()
        mock_base = _make_mock_base_model()
        mock_peft = _make_mock_peft_model()

        with (
            patch("keel.model.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch("keel.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
            patch("keel.model.get_peft_model", return_value=mock_peft),
            patch("keel.model.prepare_model_for_kbit_training", return_value=mock_base),
            patch("keel.model.BitsAndBytesConfig") as mock_bnb,
        ):
            from keel.model import Model

            m = Model(
                "test/model",
                strategy="qlora",
                load_in_4bit=True,
                snapshot_dir=tmp_snapshot_dir,
            )
            assert m.strategy == "qlora"
            mock_bnb.assert_called_once()

    def test_4bit_and_8bit_together_raises(self, tmp_snapshot_dir: Path) -> None:
        mock_tok = _make_mock_tokenizer()
        mock_base = _make_mock_base_model()
        mock_peft = _make_mock_peft_model()

        with (
            patch("keel.model.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch("keel.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
            patch("keel.model.get_peft_model", return_value=mock_peft),
            patch("keel.model.BitsAndBytesConfig"),
        ):
            from keel.model import KeelError, Model

            with pytest.raises(KeelError, match="Cannot use load_in_4bit and load_in_8bit"):
                Model(
                    "test/model",
                    load_in_4bit=True,
                    load_in_8bit=True,
                    snapshot_dir=tmp_snapshot_dir,
                )

    def test_invalid_strategy_still_raises(self, tmp_snapshot_dir: Path) -> None:
        mock_tok = _make_mock_tokenizer()
        mock_base = _make_mock_base_model()
        mock_peft = _make_mock_peft_model()

        with (
            patch("keel.model.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch("keel.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
            patch("keel.model.get_peft_model", return_value=mock_peft),
        ):
            from keel.model import KeelError, Model

            with pytest.raises(KeelError, match="strategy"):
                Model("test/model", strategy="full", snapshot_dir=tmp_snapshot_dir)


class TestResumeTraining:
    def test_resume_true_with_checkpoint_passes_path(
        self, patched_model, tmp_path: Path
    ) -> None:
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(json.dumps({"text": "hi"}) + "\n")

        # Create a fake checkpoint directory that learn() will find
        run_dir = Path.home() / ".keel" / "runs" / "test--model"
        checkpoint = run_dir / "checkpoint-10"
        checkpoint.mkdir(parents=True, exist_ok=True)

        mock_trainer = MagicMock()
        with (
            patch("keel.model.load_dataset") as mock_ds,
            patch("keel.model.Trainer", return_value=mock_trainer),
            patch("keel.model.TrainingArguments"),
            patch("keel.model.DataCollatorForLanguageModeling"),
        ):
            mock_dataset = MagicMock()
            mock_dataset.column_names = ["text"]
            mock_dataset.map.return_value = mock_dataset
            mock_ds.return_value = mock_dataset

            patched_model.learn(str(data_file), epochs=1, resume=True)
            call_kwargs = mock_trainer.train.call_args
            assert call_kwargs.kwargs.get("resume_from_checkpoint") == str(checkpoint)

        # Cleanup
        checkpoint.rmdir()

    def test_resume_true_no_checkpoint_starts_fresh(
        self, patched_model, tmp_path: Path
    ) -> None:
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(json.dumps({"text": "hi"}) + "\n")

        mock_trainer = MagicMock()
        with (
            patch("keel.model.load_dataset") as mock_ds,
            patch("keel.model.Trainer", return_value=mock_trainer),
            patch("keel.model.TrainingArguments"),
            patch("keel.model.DataCollatorForLanguageModeling"),
            patch("keel.model.Path.glob", return_value=iter([])),
        ):
            mock_dataset = MagicMock()
            mock_dataset.column_names = ["text"]
            mock_dataset.map.return_value = mock_dataset
            mock_ds.return_value = mock_dataset

            patched_model.learn(str(data_file), epochs=1, resume=True)
            call_kwargs = mock_trainer.train.call_args
            assert call_kwargs.kwargs.get("resume_from_checkpoint") is None

    def test_resume_false_never_passes_checkpoint(
        self, patched_model, tmp_path: Path
    ) -> None:
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(json.dumps({"text": "hi"}) + "\n")

        mock_trainer = MagicMock()
        with (
            patch("keel.model.load_dataset") as mock_ds,
            patch("keel.model.Trainer", return_value=mock_trainer),
            patch("keel.model.TrainingArguments"),
            patch("keel.model.DataCollatorForLanguageModeling"),
        ):
            mock_dataset = MagicMock()
            mock_dataset.column_names = ["text"]
            mock_dataset.map.return_value = mock_dataset
            mock_ds.return_value = mock_dataset

            patched_model.learn(str(data_file), epochs=1, resume=False)
            call_kwargs = mock_trainer.train.call_args
            assert call_kwargs.kwargs.get("resume_from_checkpoint") is None


class TestLoraTargets:
    def test_llama_targets(self) -> None:
        from keel.model import Model

        targets = Model._lora_targets("meta-llama/Llama-3.2-1B")
        assert "q_proj" in targets
        assert "k_proj" in targets

    def test_gpt2_targets(self) -> None:
        from keel.model import Model

        targets = Model._lora_targets("gpt2")
        assert "c_attn" in targets

    def test_unknown_model_uses_default(self) -> None:
        from keel.model import Model

        targets = Model._lora_targets("some-unknown-model-xyz")
        assert targets == ["q_proj", "v_proj"]
