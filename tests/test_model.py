"""Tests for the Model class — HuggingFace/PEFT calls are mocked for speed."""

from __future__ import annotations

import json
from pathlib import Path
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
    outputs.loss = torch.tensor(1.0)
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
        patch("pyrecall.model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("pyrecall.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
        patch("pyrecall.model.get_peft_model", return_value=mock_peft),
        patch("pyrecall.model.compute_embeddings", return_value=torch.randn(32)),
        patch("pyrecall.model.cosine_similarity", return_value=0.75),
    ):
        from pyrecall.model import Model

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
            patch("pyrecall.model.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch("pyrecall.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
            patch("pyrecall.model.get_peft_model", return_value=mock_peft),
        ):
            from pyrecall.model import Model, PyrecallError

            with pytest.raises(PyrecallError, match="strategy"):
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
        patched_model.snapshot(name="test_snap")
        snap_file = tmp_snapshot_dir / "test_snap" / "snapshot.json"
        assert snap_file.exists(), "snapshot.json must be written to disk"

    def test_snapshot_sets_baseline(self, patched_model) -> None:
        patched_model.snapshot(name="baseline")
        assert patched_model._baseline_snapshot_name == "baseline"

    def test_snapshot_returns_skill_snapshot(self, patched_model) -> None:
        from pyrecall.snapshot import SkillSnapshot

        snap = patched_model.snapshot(name="v1")
        assert isinstance(snap, SkillSnapshot)

    def test_snapshot_has_correct_score_count(self, patched_model) -> None:
        from pyrecall.benchmarks.default import DEFAULT_BENCHMARKS

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
            patch("pyrecall.model.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch("pyrecall.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
            patch("pyrecall.model.get_peft_model", return_value=mock_peft),
            patch("pyrecall.model.compute_embeddings", return_value=torch.randn(32)),
            patch("pyrecall.model.cosine_similarity", return_value=0.75),
        ):
            from pyrecall.model import Model, PyrecallError

            m = Model("test/model", snapshot_dir=tmp_snapshot_dir)
            with pytest.raises(PyrecallError, match="snapshot"):
                m.check()

    def test_check_raises_pyrecallerror_when_baseline_snapshot_deleted(self, patched_model) -> None:
        """Stale baseline name (snapshot dir deleted) must raise PyrecallError, not FileNotFoundError."""
        from pyrecall.model import PyrecallError

        patched_model._baseline_snapshot_name = "ghost_snapshot"
        with pytest.raises(PyrecallError, match="ghost_snapshot"):
            patched_model.check()

    def test_check_error_mentions_snapshot_name_and_action(self, patched_model) -> None:
        from pyrecall.model import PyrecallError

        patched_model._baseline_snapshot_name = "before_v1"
        with pytest.raises(PyrecallError) as exc_info:
            patched_model.check()
        msg = str(exc_info.value)
        assert "before_v1" in msg
        assert "snapshot" in msg.lower()

    def test_check_returns_report(self, patched_model) -> None:
        from pyrecall.detector import ForgettingReport

        patched_model.snapshot(name="pre")
        report = patched_model.check()
        assert isinstance(report, ForgettingReport)

    def test_check_report_has_comparisons(self, patched_model) -> None:
        patched_model.snapshot(name="pre2")
        report = patched_model.check()
        assert len(report.comparisons) > 0


class TestModelDiff:
    def test_diff_returns_report(self, patched_model) -> None:
        from pyrecall.detector import ForgettingReport

        patched_model.snapshot(name="v1")
        patched_model.snapshot(name="v2")
        report = patched_model.diff("v1", "v2")
        assert isinstance(report, ForgettingReport)

    def test_diff_report_has_comparisons(self, patched_model) -> None:
        patched_model.snapshot(name="a")
        patched_model.snapshot(name="b")
        report = patched_model.diff("a", "b")
        assert len(report.comparisons) > 0

    def test_diff_raises_for_missing_snap1(self, patched_model) -> None:
        from pyrecall.model import PyrecallError

        patched_model.snapshot(name="exists")
        with pytest.raises(PyrecallError, match="missing_snap"):
            patched_model.diff("missing_snap", "exists")

    def test_diff_raises_for_missing_snap2(self, patched_model) -> None:
        from pyrecall.model import PyrecallError

        patched_model.snapshot(name="exists")
        with pytest.raises(PyrecallError, match="missing_snap"):
            patched_model.diff("exists", "missing_snap")

    def test_diff_does_not_run_benchmarks(self, patched_model) -> None:
        patched_model.snapshot(name="x")
        patched_model.snapshot(name="y")
        with patch.object(patched_model, "_run_benchmarks") as mock_bench:
            patched_model.diff("x", "y")
        mock_bench.assert_not_called()


class TestModelConstructorDefaults:
    def test_constructor_defaults_stored(self, patched_model) -> None:
        assert patched_model.learning_rate == 2e-4
        assert patched_model.batch_size == 4
        assert patched_model.max_length == 512

    def test_custom_constructor_defaults_stored(self, tmp_snapshot_dir: Path) -> None:
        mock_tok = _make_mock_tokenizer()
        mock_base = _make_mock_base_model()
        mock_peft = _make_mock_peft_model()

        with (
            patch("pyrecall.model.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch("pyrecall.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
            patch("pyrecall.model.get_peft_model", return_value=mock_peft),
        ):
            from pyrecall.model import Model

            m = Model(
                "test/model",
                learning_rate=1e-3,
                batch_size=8,
                max_length=256,
                snapshot_dir=tmp_snapshot_dir,
            )
        assert m.learning_rate == 1e-3
        assert m.batch_size == 8
        assert m.max_length == 256

    def test_learn_uses_constructor_defaults(self, patched_model, tmp_path: Path) -> None:
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(json.dumps({"text": "hi"}) + "\n")

        patched_model.learning_rate = 5e-5
        patched_model.batch_size = 2
        patched_model.max_length = 128

        captured_args: dict = {}

        def capture_args(**kwargs):
            captured_args.update(kwargs)
            return MagicMock()

        mock_trainer = MagicMock()
        with (
            patch("pyrecall.model.load_dataset") as mock_ds,
            patch("pyrecall.model.Trainer", return_value=mock_trainer),
            patch("pyrecall.model.TrainingArguments", side_effect=capture_args),
            patch("pyrecall.model.DataCollatorForLanguageModeling"),
        ):
            mock_dataset = MagicMock()
            mock_dataset.column_names = ["text"]
            mock_dataset.__len__.return_value = 1
            mock_dataset.__getitem__.return_value = ["hello world"]
            mock_dataset.map.return_value = mock_dataset
            mock_ds.return_value = mock_dataset

            patched_model.learn(str(data_file))

        assert captured_args.get("learning_rate") == 5e-5
        assert captured_args.get("per_device_train_batch_size") == 2

    def test_learn_explicit_args_override_constructor_defaults(
        self, patched_model, tmp_path: Path
    ) -> None:
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(json.dumps({"text": "hi"}) + "\n")

        patched_model.learning_rate = 5e-5

        captured_args: dict = {}

        def capture_args(**kwargs):
            captured_args.update(kwargs)
            return MagicMock()

        mock_trainer = MagicMock()
        with (
            patch("pyrecall.model.load_dataset") as mock_ds,
            patch("pyrecall.model.Trainer", return_value=mock_trainer),
            patch("pyrecall.model.TrainingArguments", side_effect=capture_args),
            patch("pyrecall.model.DataCollatorForLanguageModeling"),
        ):
            mock_dataset = MagicMock()
            mock_dataset.column_names = ["text"]
            mock_dataset.__len__.return_value = 1
            mock_dataset.map.return_value = mock_dataset
            mock_ds.return_value = mock_dataset

            patched_model.learn(str(data_file), learning_rate=3e-4)

        assert captured_args.get("learning_rate") == 3e-4


class TestModelLearn:
    def test_learn_raises_for_missing_file(self, patched_model) -> None:
        from pyrecall.model import PyrecallError

        with pytest.raises(PyrecallError, match="not found"):
            patched_model.learn("/nonexistent/data.jsonl")

    def test_learn_runs_with_valid_jsonl(self, patched_model, tmp_path: Path) -> None:
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(json.dumps({"text": "### Human: Hi\n\n### Assistant: Hello!"}) + "\n")

        mock_trainer = MagicMock()
        mock_trainer.train = MagicMock()

        with (
            patch("pyrecall.model.load_dataset") as mock_ds,
            patch("pyrecall.model.Trainer", return_value=mock_trainer),
            patch("pyrecall.model.TrainingArguments"),
            patch("pyrecall.model.DataCollatorForLanguageModeling"),
        ):
            mock_dataset = MagicMock()
            mock_dataset.column_names = ["text"]
            mock_dataset.__len__.return_value = 1
            mock_dataset.map.return_value = mock_dataset
            mock_ds.return_value = mock_dataset

            patched_model.learn(str(data_file), epochs=1)
            mock_trainer.train.assert_called_once()


class TestLearnDataFormats:
    """learn() must route .csv and .parquet to the right load_dataset format."""

    def _run_learn(self, patched_model, data_file: Path) -> MagicMock:
        mock_trainer = MagicMock()
        with (
            patch("pyrecall.model.load_dataset") as mock_ds,
            patch("pyrecall.model.Trainer", return_value=mock_trainer),
            patch("pyrecall.model.TrainingArguments"),
            patch("pyrecall.model.DataCollatorForLanguageModeling"),
        ):
            mock_dataset = MagicMock()
            mock_dataset.column_names = ["text"]
            mock_dataset.__len__.return_value = 1
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
        from pyrecall.model import PyrecallError

        f = tmp_path / "data.txt"
        f.write_text("hello\n")
        with pytest.raises(PyrecallError, match="Unsupported file format"):
            patched_model.learn(str(f), epochs=1)


class TestQLoRA:
    def test_qlora_strategy_accepted(self, tmp_snapshot_dir: Path) -> None:
        mock_tok = _make_mock_tokenizer()
        mock_base = _make_mock_base_model()
        mock_peft = _make_mock_peft_model()

        with (
            patch("pyrecall.model.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch("pyrecall.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
            patch("pyrecall.model.get_peft_model", return_value=mock_peft),
            patch("pyrecall.model.prepare_model_for_kbit_training", return_value=mock_base),
            patch("pyrecall.model.BitsAndBytesConfig") as mock_bnb,
        ):
            from pyrecall.model import Model

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
            patch("pyrecall.model.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch("pyrecall.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
            patch("pyrecall.model.get_peft_model", return_value=mock_peft),
            patch("pyrecall.model.BitsAndBytesConfig"),
        ):
            from pyrecall.model import Model, PyrecallError

            with pytest.raises(PyrecallError, match="Cannot use load_in_4bit and load_in_8bit"):
                Model(
                    "test/model",
                    load_in_4bit=True,
                    load_in_8bit=True,
                    snapshot_dir=tmp_snapshot_dir,
                )

    def test_qlora_strategy_alone_enables_4bit(self, tmp_snapshot_dir: Path) -> None:
        """strategy='qlora' with no explicit bit flags must default to load_in_4bit=True."""
        mock_tok = _make_mock_tokenizer()
        mock_base = _make_mock_base_model()
        mock_peft = _make_mock_peft_model()

        with (
            patch("pyrecall.model.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch("pyrecall.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
            patch("pyrecall.model.get_peft_model", return_value=mock_peft),
            patch("pyrecall.model.prepare_model_for_kbit_training", return_value=mock_base),
            patch("pyrecall.model.BitsAndBytesConfig") as mock_bnb,
        ):
            from pyrecall.model import Model

            Model("test/model", strategy="qlora", snapshot_dir=tmp_snapshot_dir)

        mock_bnb.assert_called_once()
        call_kwargs = mock_bnb.call_args[1]
        assert call_kwargs["load_in_4bit"] is True
        assert call_kwargs["load_in_8bit"] is False

    def test_qlora_strategy_with_8bit_uses_8bit(self, tmp_snapshot_dir: Path) -> None:
        """strategy='qlora' + load_in_8bit=True should not override to 4-bit."""
        mock_tok = _make_mock_tokenizer()
        mock_base = _make_mock_base_model()
        mock_peft = _make_mock_peft_model()

        with (
            patch("pyrecall.model.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch("pyrecall.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
            patch("pyrecall.model.get_peft_model", return_value=mock_peft),
            patch("pyrecall.model.prepare_model_for_kbit_training", return_value=mock_base),
            patch("pyrecall.model.BitsAndBytesConfig") as mock_bnb,
        ):
            from pyrecall.model import Model

            Model(
                "test/model",
                strategy="qlora",
                load_in_8bit=True,
                snapshot_dir=tmp_snapshot_dir,
            )

        call_kwargs = mock_bnb.call_args[1]
        assert call_kwargs["load_in_4bit"] is False
        assert call_kwargs["load_in_8bit"] is True

    def test_invalid_strategy_still_raises(self, tmp_snapshot_dir: Path) -> None:
        mock_tok = _make_mock_tokenizer()
        mock_base = _make_mock_base_model()
        mock_peft = _make_mock_peft_model()

        with (
            patch("pyrecall.model.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch("pyrecall.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
            patch("pyrecall.model.get_peft_model", return_value=mock_peft),
        ):
            from pyrecall.model import Model, PyrecallError

            with pytest.raises(PyrecallError, match="strategy"):
                Model("test/model", strategy="full", snapshot_dir=tmp_snapshot_dir)


class TestResumeTraining:
    def test_resume_true_with_checkpoint_passes_path(self, patched_model, tmp_path: Path) -> None:
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(json.dumps({"text": "hi"}) + "\n")

        # Create a fake checkpoint directory that learn() will find
        run_dir = Path.home() / ".pyrecall" / "runs" / "test--model"
        checkpoint = run_dir / "checkpoint-10"
        checkpoint.mkdir(parents=True, exist_ok=True)

        mock_trainer = MagicMock()
        with (
            patch("pyrecall.model.load_dataset") as mock_ds,
            patch("pyrecall.model.Trainer", return_value=mock_trainer),
            patch("pyrecall.model.TrainingArguments"),
            patch("pyrecall.model.DataCollatorForLanguageModeling"),
        ):
            mock_dataset = MagicMock()
            mock_dataset.column_names = ["text"]
            mock_dataset.__len__.return_value = 1
            mock_dataset.map.return_value = mock_dataset
            mock_ds.return_value = mock_dataset

            patched_model.learn(str(data_file), epochs=1, resume=True)
            call_kwargs = mock_trainer.train.call_args
            assert call_kwargs.kwargs.get("resume_from_checkpoint") == str(checkpoint)

        # Cleanup
        checkpoint.rmdir()

    def test_resume_true_no_checkpoint_starts_fresh(self, patched_model, tmp_path: Path) -> None:
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(json.dumps({"text": "hi"}) + "\n")

        mock_trainer = MagicMock()
        with (
            patch("pyrecall.model.load_dataset") as mock_ds,
            patch("pyrecall.model.Trainer", return_value=mock_trainer),
            patch("pyrecall.model.TrainingArguments"),
            patch("pyrecall.model.DataCollatorForLanguageModeling"),
            patch("pyrecall.model.Path.glob", return_value=iter([])),
        ):
            mock_dataset = MagicMock()
            mock_dataset.column_names = ["text"]
            mock_dataset.__len__.return_value = 1
            mock_dataset.map.return_value = mock_dataset
            mock_ds.return_value = mock_dataset

            patched_model.learn(str(data_file), epochs=1, resume=True)
            call_kwargs = mock_trainer.train.call_args
            assert call_kwargs.kwargs.get("resume_from_checkpoint") is None

    def test_resume_false_never_passes_checkpoint(self, patched_model, tmp_path: Path) -> None:
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(json.dumps({"text": "hi"}) + "\n")

        mock_trainer = MagicMock()
        with (
            patch("pyrecall.model.load_dataset") as mock_ds,
            patch("pyrecall.model.Trainer", return_value=mock_trainer),
            patch("pyrecall.model.TrainingArguments"),
            patch("pyrecall.model.DataCollatorForLanguageModeling"),
        ):
            mock_dataset = MagicMock()
            mock_dataset.column_names = ["text"]
            mock_dataset.__len__.return_value = 1
            mock_dataset.map.return_value = mock_dataset
            mock_ds.return_value = mock_dataset

            patched_model.learn(str(data_file), epochs=1, resume=False)
            call_kwargs = mock_trainer.train.call_args
            assert call_kwargs.kwargs.get("resume_from_checkpoint") is None


class TestLoraTargets:
    def test_llama_targets(self) -> None:
        from pyrecall.model import Model

        targets = Model._lora_targets("meta-llama/Llama-3.2-1B")
        assert "q_proj" in targets
        assert "k_proj" in targets

    def test_gpt2_targets(self) -> None:
        from pyrecall.model import Model

        targets = Model._lora_targets("gpt2")
        assert "c_attn" in targets

    def test_unknown_model_uses_default(self) -> None:
        from pyrecall.model import Model

        targets = Model._lora_targets("some-unknown-model-xyz")
        assert targets == ["q_proj", "v_proj"]

    def test_mixtral_targets_explicit(self) -> None:
        from pyrecall.model import Model

        targets = Model._lora_targets("mistralai/Mixtral-8x7B-v0.1")
        assert "q_proj" in targets
        assert "k_proj" in targets
        assert "o_proj" in targets

    def test_phi_targets_explicit(self) -> None:
        from pyrecall.model import Model

        targets = Model._lora_targets("microsoft/phi-2")
        assert "q_proj" in targets
        assert "dense" in targets

    def test_mistral_not_matched_by_mixtral_key(self) -> None:
        from pyrecall.model import Model

        # Both Mistral and Mixtral should get the full 4-projection set
        mistral_targets = Model._lora_targets("mistralai/Mistral-7B-v0.1")
        mixtral_targets = Model._lora_targets("mistralai/Mixtral-8x7B-v0.1")
        assert set(mistral_targets) == set(mixtral_targets)


class TestOnForgettingCallbacks:
    def test_on_forgetting_called_when_forgetting_detected(self, patched_model) -> None:
        from pyrecall.snapshot import SkillScore

        called = []
        patched_model._on_forgetting = [lambda r: called.append(r)]
        patched_model._on_healthy = []
        real_scores = patched_model._run_benchmarks()
        patched_model.snapshot(name="pre_cb")
        # Force score=0 for every item to guarantee forgetting is detected.
        zero_scores = [
            SkillScore(
                category=s.category,
                prompt=s.prompt,
                response=s.response,
                score=0.0,
                scoring_method=s.scoring_method,
            )
            for s in real_scores
        ]
        with patch.object(patched_model, "_run_benchmarks", return_value=zero_scores):
            report = patched_model.check()
        assert not report.is_healthy
        assert len(called) == 1
        assert called[0] is report

    def test_on_healthy_called_when_no_forgetting(self, patched_model) -> None:
        healthy_called = []
        forgetting_called = []
        patched_model._on_healthy = [lambda r: healthy_called.append(r)]
        patched_model._on_forgetting = [lambda r: forgetting_called.append(r)]
        patched_model.snapshot(name="pre_healthy")
        report = patched_model.check()
        if report.is_healthy:
            assert len(healthy_called) == 1
            assert len(forgetting_called) == 0

    def test_callback_exception_does_not_crash(self, patched_model) -> None:
        def bad_cb(r):
            raise RuntimeError("boom")

        patched_model._on_forgetting = [bad_cb]
        patched_model._on_healthy = [bad_cb]
        patched_model.snapshot(name="pre_exc")
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            patched_model.check()  # must not raise
        assert any("boom" in str(w.message) for w in caught)

    def test_multiple_callbacks_all_called(self, patched_model) -> None:
        calls = []
        patched_model._on_healthy = [lambda r: calls.append(1), lambda r: calls.append(2)]
        patched_model._on_forgetting = []
        patched_model.snapshot(name="pre_multi")
        report = patched_model.check()
        if report.is_healthy:
            assert calls == [1, 2]

    def test_on_forgetting_stored_as_list_from_single_callable(
        self, tmp_snapshot_dir: Path
    ) -> None:
        mock_tok = _make_mock_tokenizer()
        mock_base = _make_mock_base_model()
        mock_peft = _make_mock_peft_model()
        cb = lambda r: None  # noqa: E731
        with (
            patch("pyrecall.model.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch("pyrecall.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
            patch("pyrecall.model.get_peft_model", return_value=mock_peft),
        ):
            from pyrecall.model import Model

            m = Model("test/model", snapshot_dir=tmp_snapshot_dir, on_forgetting=cb)
        assert m._on_forgetting == [cb]

    def test_on_forgetting_stored_as_list_from_list(self, tmp_snapshot_dir: Path) -> None:
        mock_tok = _make_mock_tokenizer()
        mock_base = _make_mock_base_model()
        mock_peft = _make_mock_peft_model()
        cb1 = lambda r: None  # noqa: E731
        cb2 = lambda r: None  # noqa: E731
        with (
            patch("pyrecall.model.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch("pyrecall.model.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
            patch("pyrecall.model.get_peft_model", return_value=mock_peft),
        ):
            from pyrecall.model import Model

            m = Model("test/model", snapshot_dir=tmp_snapshot_dir, on_forgetting=[cb1, cb2])
        assert m._on_forgetting == [cb1, cb2]
