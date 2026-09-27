# pyrecall

[![PyPI version](https://img.shields.io/pypi/v/pyrecall.svg)](https://pypi.org/project/pyrecall/)
[![CI](https://github.com/Pyrecall/Pyrecall/actions/workflows/ci.yml/badge.svg)](https://github.com/Pyrecall/Pyrecall/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pyrecall?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/pyrecall)

**What is pyrecall?**

pyrecall helps mitigate catastrophic forgetting while fine tuning. Fine tuning is one of the most relevant things out there at this moment, but it can set some drawbacks. While fine tuning on new data, some previous knowledge may get lost. That’s what pyrecall helps detect.

---

## Insert the following into your terminal

```bash
pip install pyrecall
```

---

## Basic API commands for pyrecall to get started

**This initializes the model(default is Qwen/Qwen2.5-1.5b-Instruct), analyzes and stores all of the models current skills, and then goes through the fine-tuning process**

```python
from pyrecall import Model

model = Model("Qwen/Qwen2.5-1.5b-Instruct")
model.snapshot("baseline")
model.learn("data.jsonl", epochs=3)
```


**This initializes the model(default is Qwen/Qwen2.5-1.5b-Instruct), analyzes and stores all of the models current skills, and then goes through the fine-tuning process**

```bash
pyrecall init
pyrecall snapshot baseline
pyrecall learn train.jsonl --snapshot-after after_training
```

Look at [examples/basic_workflow.py](examples/basic_workflow.py) for an idea of how the workflow works

## How the snapshots work

Benchmarks 20, 90, or 180 prompts across 9 categories using log-likelihood scoring. After the model finishes training, you can use the `check` command to find any differences in the new model after fine-tuning. It will flag any benchmark categories that drop past your threshold. Each snapshot stores a LoRA adapter or a QLoRa adapter.

Any LM on HuggingFace Hub is supported (the default model is Qwen/Qwen2.5-1.5b-Instruct).

---

## Docs

Access our Pyrecall's documentation here

**[pyrecall.github.io/Pyrecall](https://pyrecall.github.io/Pyrecall/)**

---

## How to Contribute

1. Join our discord group!(https://discord.gg/32dYfhZwrG)
2. Create a forked repository of this repository
3. Check the issues tab to see what needs to be done and pick one
4. Comment on that issue to notify us that you're working on that issue, or message us on discord
5. Clone your forked repository into your code editor, and push once you're done
6. Create a pull request on the original repository for the moderators to review and approve or reject
7. Do it all again!

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md) for more.

MIT — [LICENSE](LICENSE)

Contributors: @Arths17, @Sid294, @shreyasgandhe (portions of this codebase were developed with AI assistance)
