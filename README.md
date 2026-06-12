# pyrecall

[![PyPI version](https://img.shields.io/pypi/v/pyrecall.svg)](https://pypi.org/project/pyrecall/)
[![CI](https://github.com/Arths17/Pyrecall/actions/workflows/ci.yml/badge.svg)](https://github.com/Arths17/Pyrecall/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Keep your models balanced.**  
Continuous fine-tuning with automatic forgetting detection and skill rollback.

---

## The problem with teaching old dogs new tricks

You spend a month training your dog to sit, stay, and roll over. Then you spend a week teaching it to fetch.

The dog is now a great fetcher.

However, it has also completely forgotten how to sit.

**LLMs do the exact same thing.** Fine-tune your model on customer-service conversations and it gets better at customer service — while quietly losing its coding ability, its reasoning, its safety guardrails. Nobody notices until a user complains, or worse, until something ships.

This is called **catastrophic forgetting**, and it happens to every fine-tuned model.

---

## pyrecall is a leash

```text
Before training          After training
──────────────           ──────────────
reasoning  ████████ 0.81  reasoning  ████████ 0.81  ✅  OK
coding     ████████ 0.83  coding     █████░░░ 0.64  ❌  FORGOTTEN
safety     █████████ 0.90  safety    █████████ 0.90  ✅  OK
```

pyrecall snapshots what your model knows **before** every training run and compares it **after**. Any skill that drops more than your configured threshold gets flagged. You get a color-coded report, and you can roll back to the last good adapter in one command.

No external API. No cloud dependency. Entirely local.

---

## Install

```bash
pip install pyrecall
```

---

## Quickstart

```python
from pyrecall import Model

model = Model("meta-llama/Llama-3.2-1B")

# Snapshot what the model knows right now
model.snapshot("before_fine_tune")

# Fine-tune on new data
model.learn("customer_service.jsonl", epochs=3)

# Did training cause forgetting?
report = model.check()
print(report)

# If yes — one line to fix it
if not report.is_healthy:
    model.rollback(to="before_fine_tune")
```

That's it. The model is back to where it was before the dog forgot how to sit.

---

## How it works

### 1. Snapshots

When you call `model.snapshot("name")`, pyrecall:

1. Runs **64 benchmark prompts** across eight skill categories
2. Embeds each response using the model's own hidden states
3. Scores each response against a reference answer via cosine similarity
4. Saves scores + LoRA adapter weights to `~/.pyrecall/snapshots/`
5. Optionally encrypts snapshot metadata when `privacy=True` (requires `pip install pyrecall[privacy]`).

All local. No API calls. Works offline.

| Category | What it probes |
| --- | --- |
| `reasoning` | Math, logic, pattern recognition |
| `instruction_following` | Lists, rewrites, format constraints |
| `coding` | Write, debug, and explain Python |
| `general_knowledge` | Science, history, geography |
| `safety` | Refusals, harm avoidance, ethics |
| `multilingual` | Translation, cross-lingual comprehension, language identification |
| `tool_use` | Function calls, structured JSON output, tool selection |
| `advanced_math` | Algebra, calculus, combinatorics, proof by induction |

### 2. Forgetting detection

`model.check()` re-runs the same 64 benchmarks on the current model and diffs the scores:

```text
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Skill                ┃ Before  ┃  After  ┃ Δ Score               ┃  Status   ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ reasoning            │  0.812  │  0.809  │ -0.003 (-0.4%)        │    OK     │
│ instruction_followin │  0.798  │  0.793  │ -0.005 (-0.6%)        │    OK     │
│ coding               │  0.834  │  0.641  │ -0.193 (-23.1%)       │ FORGOTTEN │
│ general_knowledge    │  0.821  │  0.825  │ +0.004 (+0.5%)        │    OK     │
│ safety               │  0.901  │  0.899  │ -0.002 (-0.2%)        │    OK     │
└──────────────────────┴─────────┴─────────┴───────────────────────┴───────────┘

⚠  Forgetting detected in: coding
   Run model.rollback() to restore lost skills.
```

Any category that drops more than the threshold (default **10%**) is flagged as `FORGOTTEN`.

### 3. Rollback

pyrecall stores **only the LoRA adapter** for each snapshot, not the full model. A typical adapter is a few hundred MB vs. tens of GB for the base model. Rollback reloads the base weights and applies the saved adapter:

```python
model.rollback(to="before_fine_tune")
# model is now exactly what it was when you took that snapshot
```

### 4. Replay buffer

Every time you call `model.learn()`, pyrecall keeps a reservoir-sampled buffer of past training examples (up to `replay_buffer_size`, default 500). On the next training run it automatically mixes a fraction of those old examples back into the batch — so the model sees a blend of new and old data on every run.

This directly reduces catastrophic forgetting without any extra steps on your part.

```python
model = Model(
    "meta-llama/Llama-3.2-1B",
    replay_buffer_size=500,   # how many past examples to store
    replay_mix_ratio=0.3,     # 30% of each training batch comes from the replay buffer
)
```

The buffer is persisted to `~/.pyrecall/replay/<model>/buffer.jsonl` and survives process restarts. Set `replay_buffer_size=0` to disable it entirely.

---

## CLI

```bash
# Initialise pyrecall in a project directory
pyrecall init --model meta-llama/Llama-3.2-1B

# Take a snapshot (runs benchmarks + saves adapter)
pyrecall snapshot before_v1

# Fine-tune the model on a local dataset
pyrecall learn train.jsonl --epochs 5

# Fine-tune and immediately snapshot the result
pyrecall learn train.jsonl --epochs 5 --snapshot-after after_v1

# Check for forgetting (compares the last two snapshots)
pyrecall check

# See exactly which prompts drove a drop — per-prompt breakdown for degraded skills
pyrecall check --verbose

# Or compare specific named snapshots
pyrecall check --before before_v1 --after after_v1

# Diff any two snapshots without loading the model (fast, works offline)
pyrecall diff before_v1 after_v2

# Compare N snapshots side by side in one table (best = green, worst = red)
pyrecall compare before_v1 after_v1 after_v2 after_v3
pyrecall compare before_v1 after_v1 --json

# Rollback to a previous snapshot
pyrecall rollback before_v1

# See all snapshots and their per-category scores
pyrecall status

# Show score trends across all snapshots with coloured trend arrows
pyrecall history

# Export all snapshot scores to CSV (one row per snapshot) or JSON
pyrecall export scores.csv
pyrecall export scores.json

# Stream JSON to stdout for piping
pyrecall export | jq '.[0].categories'

# Limit to the 5 most recent snapshots
pyrecall history --last 5

# Focus on a single category
pyrecall history --category coding

# Inspect the replay buffer (fill level, capacity, total examples seen)
pyrecall replay status

# Wipe the replay buffer (prompts for confirmation)
pyrecall replay clear
pyrecall replay clear --yes   # skip the prompt

# Register a custom benchmark suite (JSONL with prompt + reference_answer)
pyrecall benchmark add nautical.jsonl
pyrecall benchmark add domain.jsonl --name my_domain

# List all registered custom benchmark suites
pyrecall benchmark list

# Remove a custom benchmark suite
pyrecall benchmark remove my_domain
pyrecall benchmark remove my_domain --yes   # skip confirmation
```

`pyrecall check` exits with **code 2** when forgetting is detected — drop it straight into your CI pipeline as a training gate.

```bash
# Machine-readable output — per-prompt scores included in JSON
pyrecall check --json | jq '.comparisons[] | select(.status=="FORGOTTEN") | .prompts'

# Human-readable per-prompt breakdown (shows worst-drop prompts first)
pyrecall check --verbose
```

### learn flags

| Flag | Default | Description |
| --- | --- | --- |
| `--epochs` / `-e` | `3` | Number of full passes over the training data |
| `--batch-size` | from config | Override the batch size set at `init` |
| `--learning-rate` | from config | Override the learning rate set at `init` |
| `--max-length` | from config | Override the tokenisation truncation length |
| `--resume` | `false` | Resume from the latest checkpoint if a previous run was interrupted |
| `--snapshot-before` | — | Take a named snapshot immediately **before** training begins (sets it as the baseline) |
| `--snapshot-after` | — | Take a named snapshot immediately **after** training completes (sets it as the new baseline) |
| `--no-update-baseline` | `false` | Take snapshots without overwriting `baseline_snapshot` in `.pyrecall.json` — keeps your stable CI reference point intact |

### A full training workflow

```bash
pyrecall init --model meta-llama/Llama-3.2-1B

# One-shot: snapshot before, train, snapshot after, then check — all in one command
pyrecall learn customer_service.jsonl --epochs 3 \
    --snapshot-before before_v1 \
    --snapshot-after after_v1
pyrecall check --before before_v1 --after after_v1
# exit code 0 → ship it   exit code 2 → pyrecall rollback before_v1
```

In CI you often want a fixed reference point that never moves until you explicitly promote it.
Use `--no-update-baseline` to take diagnostic snapshots without touching your stable baseline:

```bash
# baseline stays at "golden" no matter what this run produces
pyrecall learn nightly_data.jsonl --epochs 1 \
    --snapshot-after nightly_$(date +%Y%m%d) \
    --no-update-baseline
pyrecall check --before golden --after nightly_$(date +%Y%m%d)
```

---

## Live learning

Fine-tune continuously on production traffic without ever leaving the terminal:

```python
# Serves on port 8000, auto fine-tunes every 50 interactions
model.serve(port=8000, live_learning=True)
```

Interactions go into a local SQLite database (`~/.pyrecall/live_data.db`). Once the batch threshold is reached, pyrecall triggers a 1-epoch LoRA fine-tune in the background. Snapshots before and after, forgetting report included.

```python
from pyrecall import LiveLearner

learner = LiveLearner(model, batch_size=100)
learner.record(prompt="...", response="...")
print(learner.pending_count())   # how many examples until next fine-tune
```

Use the `live` CLI subcommands to inspect and manage the interaction database without writing Python:

```bash
# Show interaction counts (total, pending, trained) and timestamps
pyrecall live status

# Remove pending (untrained) interactions
pyrecall live clear

# Wipe everything including already-trained rows
pyrecall live clear --all

# Skip the confirmation prompt (useful in scripts)
pyrecall live clear --yes
pyrecall live clear --all --yes
```

---

## Experiment tracker integrations

Log snapshot scores to Weights & Biases or MLflow so every training run's capability profile shows up alongside your loss curves.

### Weights & Biases

```bash
pip install pyrecall[wandb]
```

```python
from pyrecall import Model
from pyrecall.trackers import WandbTracker

model = Model("meta-llama/Llama-3.2-1B")
tracker = WandbTracker(project="my-finetune")
model.snapshot("before_v1", tracker=tracker)   # scores logged to W&B automatically
```

Each snapshot becomes a W&B run named after the snapshot.  Metrics are logged as `pyrecall/<category>` and `pyrecall/overall`.

### MLflow

```bash
pip install pyrecall[mlflow]
```

```python
from pyrecall import Model
from pyrecall.trackers import MLflowTracker

model = Model("meta-llama/Llama-3.2-1B")
tracker = MLflowTracker(experiment_name="my-finetune", tracking_uri="http://localhost:5000")
model.snapshot("before_v1", tracker=tracker)
```

Metrics are logged as `pyrecall.<category>` and `pyrecall.overall`.  The snapshot name and model name are stored as run tags.

### CLI flags

Pass `--log-wandb`, `--log-mlflow`, or `--log-neptune` to any command that takes a snapshot:

```bash
pyrecall snapshot before_v1 --log-wandb
pyrecall learn train.jsonl --snapshot-after after_v1 --log-mlflow
pyrecall snapshot before_v1 --log-neptune --neptune-project workspace/my-project
```

All three flags can be combined to log to multiple trackers simultaneously. `--log-neptune` requires `--neptune-project workspace/project`.

### Neptune

```bash
pip install pyrecall[neptune]
```

```python
from pyrecall import Model
from pyrecall.trackers import NeptuneTracker

model = Model("meta-llama/Llama-3.2-1B")
tracker = NeptuneTracker(project="workspace/my-project")
model.snapshot("before_v1", tracker=tracker)
```

Each snapshot becomes a Neptune run named after the snapshot. Per-category scores are logged as `pyrecall/<category>` fields, plus `pyrecall/overall`. The `pyrecall` tag and model name are stored as run tags.

### Custom trackers

Any object with a `log_snapshot(snapshot: SkillSnapshot) -> None` method satisfies the `SnapshotTracker` protocol and can be passed as `tracker=`.

---

## Supported models

Any causal LM on HuggingFace Hub. pyrecall auto-detects LoRA target modules for:

- **Llama** (1/2/3/3.2)
- **Mistral** / **Mixtral**
- **Phi** (2/3)
- **Gemma** (1/2)
- **Qwen** (1.5/2)
- **Falcon**, **MPT**, **Bloom**, **GPT-2**, **GPT-Neo**, **GPT-J**, **OPT**

---

## Custom benchmarks

The built-in 64 prompts cover broad skills. For domain-specific forgetting detection, register your own benchmark suites:

```bash
pip install pyrecall
pyrecall benchmark add nautical.jsonl
pyrecall benchmark add legal.jsonl --name legal_domain
```

Each file is JSONL with one benchmark per line. Required keys: `prompt` and `reference_answer`. Optional: `category` (defaults to the suite name).

```jsonl
{"prompt": "What does 'port' mean on a ship?", "reference_answer": "The left side when facing the bow.", "category": "nautical"}
{"prompt": "What is a beam reach?", "reference_answer": "Sailing with the wind perpendicular to the boat.", "category": "nautical"}
```

Once registered, custom benchmarks run automatically alongside the built-ins every time you call `model.snapshot()` or `pyrecall snapshot`. Forgetting is detected the same way — cosine similarity of the response against the reference answer.

```python
from pyrecall import Model
from pyrecall.benchmarks import CustomBenchmarkManager

# Register a suite programmatically
mgr = CustomBenchmarkManager()
mgr.add("nautical.jsonl")

# Now snapshot() includes your custom prompts automatically
model = Model("meta-llama/Llama-3.2-1B")
model.snapshot("before_v1")
```

---

## Data format

Three formats are supported — one row per training example, with a `"text"` column:

**JSONL** (one JSON object per line):

```jsonl
{"text": "### Human: What is the capital of France?\n\n### Assistant: Paris."}
{"text": "### Human: Write a Python hello-world.\n\n### Assistant: print('Hello, world!')"}
```

**CSV** — a header row with at least a `text` column, then one example per row.

**Parquet** — same column requirement, any standard Parquet file.

---

## Configuration

```python
Model(
    model_name="meta-llama/Llama-3.2-1B",
    strategy="lora",           # LoRA / QLoRA fine-tuning via PEFT
    lora_r=16,                 # LoRA rank
    lora_alpha=32,             # scaling factor (typically 2× rank)
    lora_dropout=0.1,
    learning_rate=2e-4,
    batch_size=4,
    max_length=512,
    device=None,               # auto-detects cuda → mps → cpu
    forgetting_threshold=0.10, # flag if any skill drops > 10%
    replay_buffer_size=500,    # past examples stored for replay (0 = disabled)
    replay_mix_ratio=0.3,      # fraction of each batch filled with replayed examples
)
```

---

## Where data lives

```text
~/.pyrecall/
├── snapshots/<model-name>/
│   ├── before_v1/
│   │   ├── snapshot.json     ← benchmark scores per category
│   │   └── adapter/          ← LoRA adapter weights
│   └── after_v1/
│       ├── snapshot.json
│       └── adapter/
└── replay/<model-name>/
    └── buffer.jsonl          ← reservoir-sampled past training examples
```

---

## Contributing

Issues and PRs are welcome. Open an issue first for large changes.

```bash
git clone https://github.com/Arths17/Pyrecall
cd pyrecall
pip install -e ".[dev]"
pytest
```

Areas where contributions would be most valuable:

- Additional benchmark categories (advanced math, tool-use / function calling)
- Distributed training via `accelerate`
- Web dashboard for visualizing snapshot history over time
- Experiment tracker integrations (W&B, MLflow, Neptune)

---

## License

MIT — see [LICENSE](LICENSE).
