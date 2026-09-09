# pyrecall

[![PyPI version](https://img.shields.io/pypi/v/pyrecall.svg)](https://pypi.org/project/pyrecall/)
[![CI](https://github.com/Pyrecall/Pyrecall/actions/workflows/ci.yml/badge.svg)](https://github.com/Pyrecall/Pyrecall/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pyrecall?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/pyrecall)

**Forgetting detection and skill rollback for fine-tuned LLMs.**

Fine-tuning always comes with risks. Every time you fine-tune, your model may lose some of the knowledge it previously had.

---

## How to Install

```bash
pip install pyrecall
```

---

## To Get Started

```python
from pyrecall import Model

model = Model("Qwen/Qwen2.5-1.5b-Instruct")
model.snapshot("baseline")
model.learn("data.jsonl", epochs=3)
```

```bash
pyrecall init
pyrecall snapshot baseline
pyrecall learn train.jsonl --snapshot-after after_training
pyrecall check
```

Look at [examples/basic_workflow.py](examples/basic_workflow.py) for an idea of how the workflow works

## How it works

Benchmarks 20, 90, or 180 prompts across 9 categories using log-likelihood scoring. After the model finishes training, you can use the `check` command to find any differences in the new model after fine-tuning. It will flag any benchmark categories that drop past your threshold. Each snapshot stores a LoRA adapter or a QLoRa adapter.

Any LM on HuggingFace Hub is supported (the default model is Qwen/Qwen2.5-1.5b-Instruct).

---

## Docs

Access our Pyrecall's documentation here

**[pyrecall.github.io/Pyrecall](https://pyrecall.github.io/Pyrecall/)**

---

## Contributing

```bash
git clone https://github.com/Pyrecall/Pyrecall
pip install -e ".[dev]"
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md) for more.

MIT — [LICENSE](LICENSE)

Contributors: @Arths17, @Sid294, @shreyasgandhe (portions of this codebase were developed with AI assistance)
