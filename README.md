# pyrecall

[![PyPI version](https://img.shields.io/pypi/v/pyrecall.svg)](https://pypi.org/project/pyrecall/)
[![CI](https://github.com/Arths17/Pyrecall/actions/workflows/ci.yml/badge.svg)](https://github.com/Arths17/Pyrecall/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pyrecall?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/pyrecall)

**Keep your models balanced.**  
Continuous fine-tuning with automatic forgetting detection and skill rollback.

---

## The problem

You fine-tune your model on customer-service conversations. It gets better at customer service — while quietly losing its coding ability, its reasoning, its safety guardrails. Nobody notices until a user complains, or worse, until something ships.

This is called **catastrophic forgetting**, and it happens to every fine-tuned model.

---

## What pyrecall does

```text
Before training          After training
──────────────           ──────────────
reasoning  ████████ 0.81  reasoning  ████████ 0.81  ✅  OK
coding     ████████ 0.83  coding     █████░░░ 0.64  ❌  FORGOTTEN
safety     █████████ 0.90  safety    █████████ 0.90  ✅  OK
```

pyrecall snapshots what your model knows **before** every training run and compares it **after**. Any skill that drops more than your configured threshold gets flagged. You get a color-coded report, and you can roll back to the last good adapter in one command.

No external API. No cloud dependency. Entirely local. Works on CPU-only hardware.

---

## Install

```bash
pip install pyrecall
```

Supports Python 3.11–3.14.

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

---

## How it works

### Snapshots

`model.snapshot("name")` runs 180 benchmark prompts across nine skill categories and scores each using **log-likelihood** — the model's own probability of generating the reference answer. Scores + LoRA adapter weights are saved to `~/.pyrecall/snapshots/`.

| Category | What it probes |
| --- | --- |
| `reasoning` | Math, logic, pattern recognition |
| `instruction_following` | Lists, rewrites, format constraints |
| `coding` | Write, debug, and explain Python |
| `general_knowledge` | Science, history, geography |
| `safety` | Refusals, harm avoidance, ethics |
| `multilingual` | Translation, cross-lingual comprehension |
| `tool_use` | Function calls, structured JSON output |
| `advanced_math` | Algebra, calculus, combinatorics |
| `long_context` | Document QA, multi-hop retrieval |

### Forgetting detection

`model.check()` re-runs the same benchmarks and diffs the scores. Each category gets a **Cohen's d** effect size so you know whether a drop is noise or a real regression. Any category that falls more than the threshold (default 10%) is flagged in `report.degraded_skills`.

### Rollback

pyrecall stores only the LoRA adapter per snapshot — a few hundred MB, not the full model. `model.rollback(to="name")` reloads the base weights and applies the saved adapter.

### Replay buffer

Every `model.learn()` call keeps a reservoir-sampled buffer of past training examples and automatically mixes them back into the next training batch, directly reducing forgetting without extra steps.

---

## CLI

```bash
# Initialise in a project directory
pyrecall init --model meta-llama/Llama-3.2-1B

# Snapshot → train → check — the core loop
pyrecall snapshot before_v1
pyrecall learn train.jsonl --epochs 5 --snapshot-after after_v1
pyrecall check --before before_v1 --after after_v1

# View all snapshots
pyrecall status
pyrecall status --json

# Roll back
pyrecall rollback before_v1

# Score trends over time
pyrecall history
pyrecall history --category coding --last 10

# Machine-readable check output
pyrecall check --json | jq '.comparisons[] | select(.status=="FORGOTTEN")'
```

`pyrecall check` exits with **code 2** when forgetting is detected — use it as a CI training gate.

For the full CLI reference see the [docs](https://pyrecall.github.io/Pyrecall/).

---

## Supported models

Any causal LM on HuggingFace Hub. LoRA target modules are auto-detected for:

- **Llama** (1/2/3/3.2)
- **Mistral** / **Mixtral**
- **Phi** (2/3)
- **Gemma** (1/2)
- **Qwen** (1.5/2/2.5)
- **Falcon**, **MPT**, **Bloom**, **GPT-2**, **GPT-Neo**, **GPT-J**, **OPT**

---

## Documentation

Full docs — CLI reference, Python API, experiment tracker integrations (W&B, MLflow, Neptune), custom benchmarks, per-category thresholds, privacy/encryption, and more:

**[pyrecall.github.io/Pyrecall](https://pyrecall.github.io/Pyrecall/)**

---

## Contributing

Issues and PRs are welcome. Open an issue first for large changes.

```bash
git clone https://github.com/Arths17/Pyrecall
cd pyrecall
pip install -e ".[dev]"
pytest
```

---

## License

MIT — see [LICENSE](LICENSE).
