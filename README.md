# keel

[![PyPI version](https://img.shields.io/pypi/v/keelfit.svg)](https://pypi.org/project/keelfit/)
[![CI](https://github.com/Arths17/keel/actions/workflows/ci.yml/badge.svg)](https://github.com/Arths17/keel/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Keep your models balanced.**  
Continuous fine-tuning with automatic forgetting detection and skill rollback.

---

## The problem with teaching old dogs new tricks

You spend a month training your dog to sit, stay, and roll over. Then you spend a week teaching it to fetch.

The dog is now a great fetcher.

It has also completely forgotten how to sit.

**LLMs do the exact same thing.** Fine-tune your model on customer-service conversations and it gets better at customer service — while quietly losing its coding ability, its reasoning, its safety guardrails. Nobody notices until a user complains, or worse, until something ships.

This is called **catastrophic forgetting**, and it happens to every fine-tuned model.

---

## keel is a leash

```text
Before training          After training
──────────────           ──────────────
reasoning  ████████ 0.81  reasoning  ████████ 0.81  ✅  OK
coding     ████████ 0.83  coding     █████░░░ 0.64  ❌  FORGOTTEN
safety     █████████ 0.90  safety    █████████ 0.90  ✅  OK
```

keel snapshots what your model knows **before** every training run and compares it **after**. Any skill that drops more than your configured threshold gets flagged. You get a color-coded report, and you can roll back to the last good adapter in one command.

No external API. No cloud dependency. Entirely local.

---

## Install

```bash
pip install keelfit
```

---

## Quickstart

```python
from keel import Model

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

When you call `model.snapshot("name")`, keel:

1. Runs **20 benchmark prompts** across five skill categories
2. Embeds each response using the model's own hidden states
3. Scores each response against a reference answer via cosine similarity
4. Saves scores + LoRA adapter weights to `~/.keel/snapshots/`

All local. No API calls. Works offline.

| Category | What it probes |
| --- | --- |
| `reasoning` | Math, logic, pattern recognition |
| `instruction_following` | Lists, rewrites, format constraints |
| `coding` | Write, debug, and explain Python |
| `general_knowledge` | Science, history, geography |
| `safety` | Refusals, harm avoidance, ethics |

### 2. Forgetting detection

`model.check()` re-runs the same 20 benchmarks on the current model and diffs the scores:

```
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

keel stores **only the LoRA adapter** for each snapshot, not the full model. A typical adapter is a few hundred MB vs. tens of GB for the base model. Rollback reloads the base weights and applies the saved adapter:

```python
model.rollback(to="before_fine_tune")
# model is now exactly what it was when you took that snapshot
```

---

## CLI

```bash
# Initialise keel in a project directory
keel init --model meta-llama/Llama-3.2-1B

# Take a snapshot (runs benchmarks + saves adapter)
keel snapshot before_v1

# Check for forgetting (compares the last two snapshots)
keel check

# Or compare specific named snapshots
keel check --before before_v1 --after after_fine_tune

# Rollback to a previous snapshot
keel rollback before_v1

# See all snapshots and their per-category scores
keel status
```

`keel check` exits with **code 2** when forgetting is detected — drop it straight into your CI pipeline as a training gate.

---

## Live learning

Fine-tune continuously on production traffic without ever leaving the terminal:

```python
# Serves on port 8000, auto fine-tunes every 50 interactions
model.serve(port=8000, live_learning=True)
```

Interactions go into a local SQLite database (`~/.keel/live_data.db`). Once the batch threshold is reached, keel triggers a 1-epoch LoRA fine-tune in the background. Snapshots before and after, forgetting report included.

```python
from keel import LiveLearner

learner = LiveLearner(model, batch_size=100)
learner.record(prompt="...", response="...")
print(learner.pending_count())   # how many examples until next fine-tune
```

---

## Supported models

Any causal LM on HuggingFace Hub. keel auto-detects LoRA target modules for:

- **Llama** (1/2/3/3.2)
- **Mistral** / **Mixtral**
- **Phi** (2/3)
- **Gemma** (1/2)
- **Qwen** (1.5/2)
- **Falcon**, **MPT**, **Bloom**, **GPT-2**, **GPT-Neo**, **GPT-J**, **OPT**

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
    forgetting_threshold=0.10  # flag if any skill drops > 10%
)
```

---

## Where snapshots live

```
~/.keel/snapshots/<model-name>/
├── before_v1/
│   ├── snapshot.json     ← benchmark scores per category
│   └── adapter/          ← LoRA adapter weights (only file needed for rollback)
└── after_fine_tune/
    ├── snapshot.json
    └── adapter/
```

---

## Contributing

Issues and PRs are welcome. Open an issue first for large changes.

```bash
git clone https://github.com/Arths17/keel
cd keel
pip install -e ".[dev]"
pytest
```

Areas where contributions would be most valuable:

- Additional benchmark categories (multilingual, advanced math, tool-use / function calling)
- QLoRA support (`load_in_4bit` / `load_in_8bit` via `bitsandbytes`)
- Distributed training via `accelerate`
- Web dashboard for visualizing snapshot history over time
- Experiment tracker integrations (W&B, MLflow, Neptune)

---

## License

MIT — see [LICENSE](LICENSE).
