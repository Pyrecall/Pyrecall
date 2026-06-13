"""Model — wraps a HuggingFace causal LM with continuous fine-tuning capabilities."""

from __future__ import annotations

import asyncio
import inspect
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset as _HFDataset
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from rich.progress import Progress, SpinnerColumn, TextColumn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .benchmarks.custom import CustomBenchmarkManager
from .benchmarks.default import DEFAULT_BENCHMARKS
from .detector import ForgettingDetector, ForgettingReport
from .replay import ReplayBuffer
from .rollback import RollbackManager
from .snapshot import SkillScore, SkillSnapshot
from .trackers import SnapshotTracker
from .utils import (
    compute_embeddings,
    compute_log_likelihood,
    console,
    cosine_similarity,
    get_logger,
    safe_model_name,
)

logger = get_logger(__name__)

# ── LoRA target-module heuristics ─────────────────────────────────────────────
# Maps a substring of the model name (lowercase) to the attention projection layers
# that LoRA should adapt. The "default" key is the fallback.
_LORA_TARGETS: dict[str, list[str]] = {
    "llama": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "mistral": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "mixtral": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "phi": ["q_proj", "v_proj", "k_proj", "dense"],
    "qwen": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "gemma": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "falcon": ["query_key_value"],
    "mpt": ["Wqkv"],
    "bloom": ["query_key_value"],
    "gpt2": ["c_attn", "c_proj"],
    "gpt-neo": ["q_proj", "v_proj"],
    "gpt-j": ["q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "default": ["q_proj", "v_proj"],
}


class PyrecallError(Exception):
    """Raised for user-facing operational errors in pyrecall."""


# Type alias for a single forgetting callback or a list of them.
ForgettingCallback = Callable[["ForgettingReport"], Any]


def _fire_callbacks(
    callbacks: list[ForgettingCallback],
    report: ForgettingReport,
) -> None:
    """Invoke each callback with *report*, handling async and exceptions gracefully."""
    for cb in callbacks:
        try:
            result = cb(report)
            if inspect.iscoroutine(result):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(result)
                except RuntimeError:
                    asyncio.run(result)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"pyrecall: callback {cb!r} raised {type(exc).__name__}: {exc}",
                stacklevel=2,
            )


class Model:
    """
    A HuggingFace causal LM wrapped for continuous fine-tuning.

    Provides four lifecycle methods that mirror git's snapshot/commit/diff/reset:

    * :meth:`snapshot`  — record current capabilities before training.
    * :meth:`learn`     — fine-tune on new data using LoRA.
    * :meth:`check`     — compare capabilities before and after to detect forgetting.
    * :meth:`rollback`  — restore a previous snapshot's adapter weights.

    Example::

        model = Model("meta-llama/Llama-3.2-1B", strategy="lora")
        model.snapshot(name="before_v1")
        model.learn("data.jsonl", epochs=3)
        report = model.check()
        if not report.is_healthy:
            model.rollback(to="before_v1")
    """

    rollback_manager: RollbackManager
    _baseline_file: Path
    _baseline_snapshot_name: str | None

    def __init__(
        self,
        model_name: str,
        strategy: str = "lora",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        device: str | None = None,
        snapshot_dir: Path | None = None,
        forgetting_threshold: float = 0.10,
        category_thresholds: dict[str, float] | None = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        max_length: int = 512,
        replay_buffer_size: int = 500,
        replay_mix_ratio: float = 0.3,
        scoring_method: str = "log_likelihood",
        on_forgetting: ForgettingCallback | list[ForgettingCallback] | None = None,
        on_healthy: ForgettingCallback | list[ForgettingCallback] | None = None,
        snapshot_compression: str = "none",
        gradient_checkpointing: bool = False,
    ) -> None:
        """
        Load *model_name* from HuggingFace Hub (or local cache) and wrap it with LoRA.

        Args:
            model_name: HuggingFace model identifier, e.g. ``"meta-llama/Llama-3.2-1B"``.
            strategy: Fine-tuning strategy — ``"lora"`` or ``"qlora"``.
            lora_r: LoRA rank (lower = fewer parameters, higher = more capacity).
            lora_alpha: LoRA scaling factor (usually 2× rank).
            lora_dropout: Dropout applied to LoRA layers.
            device: ``"cuda"``, ``"cpu"``, or ``"mps"``. Auto-detected when None.
            snapshot_dir: Override the default ``~/.pyrecall/snapshots/<model>`` path.
            forgetting_threshold: Score drop fraction that counts as forgetting (0–1).
            load_in_4bit: Load base model in 4-bit (QLoRA). Requires ``bitsandbytes``.
            load_in_8bit: Load base model in 8-bit. Requires ``bitsandbytes``.
            learning_rate: Default AdamW learning rate used by :meth:`learn`.
            batch_size: Default per-device training batch size used by :meth:`learn`.
            max_length: Default tokenisation truncation length used by :meth:`learn`.
            replay_buffer_size: Maximum number of past training examples to retain in
                the replay buffer. Set to 0 to disable the buffer entirely.
            replay_mix_ratio: Fraction of each training batch to fill with replayed
                examples, e.g. 0.3 means 30 % of examples come from the buffer.
            on_forgetting: Callable (or list of callables) invoked with the
                :class:`~pyrecall.detector.ForgettingReport` when forgetting is detected.
                Sync and async callables are both supported. Exceptions are caught and
                surfaced as warnings so they never crash the training run.
            on_healthy: Same as *on_forgetting* but invoked when no forgetting is detected.
        """
        if strategy not in ("lora", "qlora"):
            raise PyrecallError(
                f"Unknown strategy '{strategy}'. "
                "pyrecall supports strategy='lora' or strategy='qlora'. "
                "Example: Model('meta-llama/Llama-3.2-1B', strategy='qlora', load_in_4bit=True)"
            )

        if scoring_method not in ("log_likelihood", "cosine"):
            raise PyrecallError(
                f"Unknown scoring_method '{scoring_method}'. "
                "Use 'log_likelihood' (recommended) or 'cosine' (legacy)."
            )
        self.rollback_manager = RollbackManager(model_name=model_name, base_dir=snapshot_dir)
        self.base_dir = snapshot_dir or (Path.home() / ".pyrecall")
        self._baseline_snapshot_name: str | None = None
        self._baseline_file = self.rollback_manager.base_dir / ".current_baseline"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # load persisted baseline if it exists
        self.model_name = model_name
        self.strategy = strategy
        self.device = device or self._best_device()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.scoring_method = scoring_method
        self.max_length = max_length
        self._replay_mix_ratio = replay_mix_ratio
        self._on_forgetting: list[ForgettingCallback] = (
            [on_forgetting] if callable(on_forgetting) else list(on_forgetting or [])
        )
        self._on_healthy: list[ForgettingCallback] = (
            [on_healthy] if callable(on_healthy) else list(on_healthy or [])
        )
        self._snapshot_compression = snapshot_compression
        self._gradient_checkpointing = gradient_checkpointing
        self.replay_buffer: ReplayBuffer | None = (
            ReplayBuffer(model_name=model_name, max_size=replay_buffer_size, base_dir=snapshot_dir)
            if replay_buffer_size > 0
            else None
        )

        console.print(f"[info]Loading {model_name} on {self.device}…[/info]")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # QLoRA: quantize base weights, keep adapters in float16.
        # strategy="qlora" implies 4-bit unless the caller explicitly requested 8-bit.
        if strategy == "qlora" and not load_in_4bit and not load_in_8bit:
            load_in_4bit = True

        bnb_config = None
        if load_in_4bit or load_in_8bit:
            if load_in_4bit and load_in_8bit:
                raise PyrecallError("Cannot use load_in_4bit and load_in_8bit together.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        dtype = torch.float16 if self.device != "cpu" else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            quantization_config=bnb_config,
            device_map="auto" if bnb_config else None,
        )

        if bnb_config:
            base = prepare_model_for_kbit_training(base)

        target_modules = self._lora_targets(model_name)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.model: Any = get_peft_model(base, lora_cfg)
        if not bnb_config:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.detector = ForgettingDetector(
            threshold=forgetting_threshold,
            category_thresholds=category_thresholds,
        )
        self.custom_benchmarks = CustomBenchmarkManager()

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        console.print(
            f"[success]✓ Model ready. "
            f"{n_trainable:,} / {n_total:,} parameters are trainable "
            f"({n_trainable / n_total:.2%}).[/success]"
        )

        if self._baseline_file.exists():
            self._baseline_snapshot_name = self._baseline_file.read_text().strip() or None

    # ── public API ─────────────────────────────────────────────────────────────

    def snapshot(
        self,
        name: str,
        tracker: SnapshotTracker | list[SnapshotTracker] | None = None,
    ) -> SkillSnapshot:
        """
        Benchmark the model and save a named capability snapshot.

        Runs all 64 default benchmarks, scores each response, saves the scores
        *and* the current LoRA adapter weights to disk so the model can be
        rolled back to this exact state later.

        Args:
            name: A human-readable label for this snapshot, e.g. ``"before_v1"``.
            tracker: An optional :class:`~pyrecall.trackers.SnapshotTracker` (or list
                of trackers) to log scores to an experiment tracker such as W&B or
                MLflow immediately after the snapshot is saved.

        Returns:
            The :class:`~pyrecall.snapshot.SkillSnapshot` that was saved.
        """
        console.print(f"[info]Taking snapshot '{name}'…[/info]")

        scores = self._run_benchmarks()
        snap = SkillSnapshot(name=name, model_name=self.model_name, scores=scores)
        self.rollback_manager.save(snap, self.model, compression=self._snapshot_compression)

        self._set_baseline(name)

        console.print(
            f"[success]✓ Snapshot '{name}' saved. "
            f"Overall score: {snap.overall_score():.3f}[/success]"
        )

        if tracker is not None:
            trackers = tracker if isinstance(tracker, list) else [tracker]
            for t in trackers:
                try:
                    t.log_snapshot(snap)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Tracker %s failed to log snapshot: %s", type(t).__name__, exc)

        return snap

    def learn(
        self,
        data_path: str,
        epochs: int = 3,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        max_length: int | None = None,
        resume: bool = False,
        gradient_checkpointing: bool | None = None,
    ) -> None:
        """
        Fine-tune the model on *data_path* using LoRA.

        *data_path* can be a ``.jsonl``, ``.csv``, or ``.parquet`` file.
        Each row must have a ``"text"`` column containing the training text.
        JSONL example line::

            {"text": "### Human: What is 2+2?\\n\\n### Assistant: 4"}

        Checkpoints are saved every 20% of an epoch to ``~/.pyrecall/runs/<model>/``.
        Pass ``resume=True`` to continue from the latest checkpoint if a previous
        run was interrupted.

        Args:
            data_path: Path to the training data file (.jsonl, .csv, or .parquet).
            epochs: Number of full passes over the training data.
            batch_size: Per-device training batch size.
            learning_rate: AdamW learning rate.
            max_length: Tokenisation truncation length.
            resume: If True, resume from the latest saved checkpoint (if one exists).
        """
        batch_size = batch_size if batch_size is not None else self.batch_size
        learning_rate = learning_rate if learning_rate is not None else self.learning_rate
        max_length = max_length if max_length is not None else self.max_length

        data_file = Path(data_path)
        if not data_file.exists():
            raise PyrecallError(
                f"Training data not found at '{data_path}'. "
                "Provide a JSONL, CSV, or Parquet file where each row has a 'text' column."
            )

        console.print(f"[info]Fine-tuning on '{data_path}' for {epochs} epoch(s)…[/info]")

        _FORMAT_MAP = {".jsonl": "json", ".json": "json", ".csv": "csv", ".parquet": "parquet"}
        fmt = _FORMAT_MAP.get(data_file.suffix.lower())
        if fmt is None:
            raise PyrecallError(
                f"Unsupported file format '{data_file.suffix}'. Use .jsonl, .csv, or .parquet."
            )

        dataset = load_dataset(fmt, data_files=str(data_file), split="train")

        # Infer the text column.
        try:
            is_empty = len(dataset) == 0
        except (TypeError, AttributeError):
            is_empty = False

        if is_empty:
            raise PyrecallError(f"Training data '{data_path}' is empty.")
        if "text" in dataset.column_names:
            text_col = "text"
        else:
            text_col = None

            # Find the first column containing non-empty text data.
            for col in dataset.column_names:
                try:
                    value = dataset[col][0]
                except (IndexError, KeyError):
                    continue

                if isinstance(value, str) and value.strip():
                    text_col = col
                    break

            if text_col is None:
                raise PyrecallError(
                    f"Could not find a usable text column in '{data_path}'.\n"
                    f"Columns found: {dataset.column_names}.\n"
                    "Rename your training text column to 'text', "
                    "or ensure at least one column contains text."
                )
        # Make sure dataset actually has usable rows
        try:
            row_count = dataset.num_rows
        except AttributeError:
            row_count = len(dataset)

        if row_count == 0:
            raise PyrecallError(f"Training data '{data_path}' is empty.")
        # Collect the raw new texts now (before any mixing) so we can add them
        # to the replay buffer after training without including replayed examples.
        new_texts: list[str] = dataset[text_col] if self.replay_buffer is not None else []

        # Mix in replay examples before tokenisation.
        if self.replay_buffer is not None and len(self.replay_buffer) > 0:
            n_replay = min(
                len(self.replay_buffer),
                max(1, int(len(dataset) * self._replay_mix_ratio)),
            )
            replay_texts = self.replay_buffer.sample(n_replay)
            replay_ds = _HFDataset.from_dict({"text": replay_texts})
            new_only = (
                dataset.select_columns([text_col]).rename_column(text_col, "text")
                if text_col != "text"
                else dataset.select_columns(["text"])
            )
            dataset = concatenate_datasets([new_only, replay_ds])
            text_col = "text"
            console.print(
                f"[info]Replay: mixed in {n_replay} past examples "
                f"({n_replay / len(dataset):.0%} of batch).[/info]"
            )

        def _tokenize(batch: dict[str, list]) -> dict:
            return self.tokenizer(
                batch[text_col],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        tokenized = dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)

        run_dir = Path.home() / ".pyrecall" / "runs" / safe_model_name(self.model_name)

        # Save a checkpoint roughly every 20% of an epoch so interrupted runs
        # can be resumed without restarting from scratch.
        save_steps = max(1, len(tokenized) // (batch_size * 5))

        use_gc = (
            gradient_checkpointing
            if gradient_checkpointing is not None
            else self._gradient_checkpointing
        )
        if use_gc:
            self.model.gradient_checkpointing_enable()

        args = TrainingArguments(
            output_dir=str(run_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=save_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=2,
            report_to="none",
            fp16=(self.device == "cuda"),
            dataloader_drop_last=False,
            gradient_checkpointing=use_gc,
        )

        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized,
            data_collator=collator,
        )

        # Find latest checkpoint when resuming
        resume_from = None
        if resume:
            checkpoints = sorted(
                run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
            )
            if checkpoints:
                resume_from = str(checkpoints[-1])
                console.print(f"[info]Resuming from checkpoint '{checkpoints[-1].name}'…[/info]")
            else:
                console.print("[warning]No checkpoint found — starting from scratch.[/warning]")

        self.model.train()
        trainer.train(resume_from_checkpoint=resume_from)
        self.model.eval()

        if self.replay_buffer is not None and new_texts:
            self.replay_buffer.add(new_texts)
            console.print(
                f"[dim]  Replay buffer updated: {len(self.replay_buffer)} / "
                f"{self.replay_buffer.max_size} examples stored.[/dim]"
            )

        console.print(f"[success]✓ Fine-tuning complete ({epochs} epoch(s)).[/success]")

    def check(self) -> ForgettingReport:
        """
        Detect forgetting by benchmarking the current model and comparing to
        the most recent snapshot.

        Must be called after at least one :meth:`snapshot` call.

        Returns:
            A :class:`~pyrecall.detector.ForgettingReport` with per-category scores
            printed to the terminal automatically.
        """
        if not self._baseline_snapshot_name:
            raise PyrecallError(
                "No baseline snapshot found.\n"
                "Call model.snapshot(name='before_v1') before fine-tuning, "
                "then call model.check() afterwards."
            )

        try:
            before = self.rollback_manager.load_snapshot(self._baseline_snapshot_name)
        except FileNotFoundError as exc:
            raise PyrecallError(
                f"Baseline snapshot '{self._baseline_snapshot_name}' not found at "
                f"'{self.rollback_manager.base_dir}'.\n"
                f"Did you run model.snapshot('{self._baseline_snapshot_name}') before training?"
            ) from exc

        console.print("[info]Running post-training benchmarks…[/info]")
        after_scores = self._run_benchmarks()
        after_name = f"{self._baseline_snapshot_name}__after"
        after = SkillSnapshot(name=after_name, model_name=self.model_name, scores=after_scores)
        after.save(self.rollback_manager.base_dir / after_name)

        report = self.detector.compare(before, after)
        report.print()
        if report.is_healthy:
            _fire_callbacks(self._on_healthy, report)
        else:
            _fire_callbacks(self._on_forgetting, report)
        return report

    def diff(self, snap1: str, snap2: str) -> ForgettingReport:
        """
        Compare two saved snapshots without running new benchmarks.

        Unlike :meth:`check`, ``diff`` does not benchmark the live model — it
        loads both snapshots from disk and diffs the stored scores directly.
        This is fast and works even if the model has been updated since the
        snapshots were taken.

        Args:
            snap1: Name of the "before" snapshot.
            snap2: Name of the "after" snapshot.

        Returns:
            A :class:`~pyrecall.detector.ForgettingReport` printed automatically.

        Example::

            report = model.diff("before_v1", "after_v2")
            if not report.is_healthy:
                model.rollback(to="before_v1")
        """
        try:
            before = self.rollback_manager.load_snapshot(snap1)
        except FileNotFoundError as exc:
            raise PyrecallError(f"Snapshot '{snap1}' not found.") from exc
        try:
            after = self.rollback_manager.load_snapshot(snap2)
        except FileNotFoundError as exc:
            raise PyrecallError(f"Snapshot '{snap2}' not found.") from exc

        report = self.detector.compare(before, after)
        report.print()
        return report

    def rollback(self, to: str) -> None:
        """
        Restore the model to the state captured in snapshot *to*.

        This:
        - reloads the base HF model
        - attaches the correct LoRA adapter
        - fully replaces the current model in memory
        - resets evaluation mode
        - updates baseline tracking safely
        """

        console.print(f"[info]Rolling back to snapshot '{to}'…[/info]")

        # ── 1. Validate snapshot before touching the model ────────────────────────
        snap = self.rollback_manager.load_snapshot(to)
        if snap.adapter_path is None:
            raise PyrecallError(f"Snapshot '{to}' is missing adapter metadata.")
        if not snap.adapter_path.exists():
            raise PyrecallError(
                f"Adapter weights not found for snapshot '{to}' at {snap.adapter_path}"
            )

        # ── 2. Load new model (do this before deleting old one) ───────────────────
        from .compress import decompressed_adapter

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=None if self.device != "cuda" else "auto",
        )
        with decompressed_adapter(snap.adapter_path, snap.adapter_compression) as adapter_dir:
            new_model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
        if self.device not in ("cuda",):
            new_model = new_model.to(self.device)

        # ── 3. Swap — only delete old model after new one is ready ────────────────
        del self.model
        self.model = new_model
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── 4. Persist baseline ───────────────────────────────────────────────────
        self._set_baseline(to)

        console.print(f"[success]✓ Rolled back to '{to}'[/success]")

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """
        Run inference and return the model's response to *prompt*.

        Args:
            prompt: The input text.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            The generated text (only the new tokens, not the prompt itself).
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)  # type: ignore[return-value]

    def serve(
        self,
        port: int = 8000,
        live_learning: bool = False,
        live_batch_size: int = 50,
    ) -> None:
        """
        Start a FastAPI inference server.

        Exposes two endpoints:

        * ``POST /generate`` — accepts ``{"prompt": "...", "max_new_tokens": 200}``,
          returns ``{"response": "...", "model": "<name>"}``.
        * ``GET /health`` — returns server status and, when *live_learning* is
          enabled, the count of pending training examples.

        Args:
            port: TCP port to bind.
            live_learning: When True, every inference request is stored and the
                model is fine-tuned automatically once *live_batch_size* interactions
                accumulate.
            live_batch_size: Number of interactions that trigger a live fine-tune run.
                Only used when *live_learning* is True.
        """
        try:
            import uvicorn
            from fastapi import FastAPI
            from fastapi.middleware.cors import CORSMiddleware
            from pydantic import BaseModel as _Base
        except ImportError as exc:
            raise PyrecallError(
                "model.serve() requires the 'serve' extra. "
                "Install it with: pip install pyrecall[serve]"
            ) from exc

        app = FastAPI(
            title="pyrecall",
            description=f"Serving {self.model_name}",
            version="0.1.0",
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        learner = None
        if live_learning:
            from .live import LiveLearner

            learner = LiveLearner(self, batch_size=live_batch_size)
            console.print(
                "[info]Live learning enabled — interactions will be collected "
                "for automatic fine-tuning.[/info]"
            )

        class GenerateRequest(_Base):
            prompt: str
            max_new_tokens: int = 200

        class GenerateResponse(_Base):
            response: str
            model: str

        @app.post("/generate", response_model=GenerateResponse)
        async def _generate(req: GenerateRequest) -> GenerateResponse:
            text = self.generate(req.prompt, req.max_new_tokens)
            if learner:
                learner.record(req.prompt, text)
            return GenerateResponse(response=text, model=self.model_name)

        @app.get("/health")
        async def _health() -> dict[str, Any]:
            info: dict[str, Any] = {
                "status": "ok",
                "model": self.model_name,
                "device": self.device,
                "baseline_snapshot": self._baseline_snapshot_name,
            }
            if learner:
                info["pending_training_examples"] = learner.pending_count()
                info["total_interactions"] = learner.total_count()
            return info

        console.print(f"[success]✓ Server starting on http://0.0.0.0:{port}[/success]")
        console.print("[dim]  POST /generate   — run inference[/dim]")
        console.print("[dim]  GET  /health     — server status[/dim]")

        uvicorn.run(app, host="0.0.0.0", port=port)

    # ── private helpers ────────────────────────────────────────────────────────
    def _set_baseline(self, name: str) -> None:
        self._baseline_snapshot_name = name
        try:
            self._baseline_file.write_text(name)
        except Exception:
            pass

    def _run_benchmarks(self) -> list[SkillScore]:
        """Run default + custom benchmarks and return SkillScore objects."""
        all_benchmarks = DEFAULT_BENCHMARKS + self.custom_benchmarks.load_all()
        scores: list[SkillScore] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Running benchmarks…", total=len(all_benchmarks))

            for bench in all_benchmarks:
                progress.update(
                    task,
                    description=(f"[{bench.category}] {bench.prompt[:55].rstrip()}…"),
                )

                if self.scoring_method == "log_likelihood":
                    score = compute_log_likelihood(
                        self.model,
                        self.tokenizer,
                        bench.prompt,
                        bench.reference_answer,
                        device=self.device,  # type: ignore[arg-type]
                        max_length=self.max_length,
                    )
                    # Still generate for human-readable storage / --verbose display
                    response = self.generate(bench.prompt)
                    if not response.strip():
                        response = "[no response]"
                else:
                    response = self.generate(bench.prompt)
                    if not response.strip():
                        response = "[no response]"
                    resp_emb = compute_embeddings(
                        self.model,
                        self.tokenizer,
                        response,
                        device=self.device,  # type: ignore[arg-type]
                    )
                    ref_emb = compute_embeddings(
                        self.model,
                        self.tokenizer,
                        bench.reference_answer,
                        device=self.device,  # type: ignore[arg-type]
                    )
                    raw_sim = cosine_similarity(resp_emb, ref_emb)
                    score = (raw_sim + 1.0) / 2.0

                scores.append(
                    SkillScore(
                        category=bench.category,
                        prompt=bench.prompt,
                        response=response,
                        score=score,
                        scoring_method=self.scoring_method,
                    )
                )

                progress.advance(task)

        return scores

    @staticmethod
    def _best_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _lora_targets(model_name: str) -> list[str]:
        name_lower = model_name.lower()
        for key, modules in _LORA_TARGETS.items():
            if key != "default" and key in name_lower:
                return modules
        return _LORA_TARGETS["default"]
