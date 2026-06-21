# pyrecall

[![PyPI version](https://img.shields.io/pypi/v/pyrecall.svg)](https://pypi.org/project/pyrecall/)
[![CI](https://github.com/Arths17/Pyrecall/actions/workflows/ci.yml/badge.svg)](https://github.com/Arths17/Pyrecall/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pyrecall?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/pyrecall)

**Forgetting detection and skill rollback for fine-tuned LLMs.**

Fine-tune on new data and your model quietly loses what it already knew — coding ability, reasoning, safety guardrails. pyrecall catches it before it ships.

---

## Install

```bash
pip install pyrecall   # Python 3.11–3.14 · CUDA, MPS, and CPU
```

---

## Quickstart

```python
from pyrecall import Model

model = Model("meta-llama/Llama-3.2-1B")
model.snapshot("before")
model.learn("data.jsonl", epochs=3)

if not model.check().is_healthy:
    model.rollback(to="before")
```

```bash
pyrecall init --model meta-llama/Llama-3.2-1B
pyrecall snapshot before_v1                        # baseline before training
pyrecall learn train.jsonl --snapshot-after after_v1  # train + snapshot in one step
pyrecall check                                     # compares last two snapshots
# exit 0 → ship   exit 2 → pyrecall rollback before_v1
```

---

## How it works

Benchmarks 180 prompts across 9 skill categories (reasoning, coding, safety, math, multilingual, and more) using log-likelihood scoring. After training, `check` diffs the scores and flags any category that drops past your threshold (default 10%). Only the LoRA adapter is stored per snapshot — a few hundred MB, not the full model.

Any causal LM on HuggingFace Hub is supported — Llama, Mistral, Phi, Gemma, Qwen, Falcon, GPT-2/Neo/J, and more. LoRA targets are auto-detected.

---

## Docs

Full CLI reference, Python API, experiment tracker integrations (W&B, MLflow, Neptune), custom benchmarks, per-category thresholds, and more:

**[pyrecall.github.io/Pyrecall](https://pyrecall.github.io/Pyrecall/)**

---

## Contributing

```bash
git clone https://github.com/Arths17/Pyrecall
pip install -e ".[dev]"
pytest
```

Open an issue before large changes. MIT — [LICENSE](LICENSE).
