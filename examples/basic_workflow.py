"""
End-to-end Pyrecall workflow: snapshot -> learn -> check -> rollback.

Run from the repo root (after `pip install -e .`):

    python examples/basic_workflow.py

By default this uses a small model (Qwen/Qwen2.5-1.5b-Instruct) and a
handful of toy training examples in sample_data.jsonl, so it can run on a
laptop CPU. Swap MODEL_NAME / DATA_PATH for your own model and dataset.
"""

from pathlib import Path

from pyrecall import Model

MODEL_NAME = "Qwen/Qwen2.5-1.5b-Instruct"
DATA_PATH = Path(__file__).parent / "sample_data.jsonl"


def main() -> None:
    model = Model(MODEL_NAME)

    # 1. Snapshot the model's current skills before touching any weights.
    model.snapshot("before")

    # 2. Fine-tune on your data. This trains a LoRA adapter on top of the
    #    frozen base model, so "before" always remains recoverable.
    model.learn(str(DATA_PATH), epochs=1)

    # 3. Benchmark the fine-tuned model against the "before" baseline and
    #    print a per-category forgetting report.
    report = model.check()

    # 4. If fine-tuning regressed the model past its forgetting threshold,
    #    roll back to the last known-good snapshot.
    if not report.is_healthy:
        print("Forgetting detected — rolling back to 'before'.")
        model.rollback(to="before")
    else:
        print("Model passed the forgetting check.")


if __name__ == "__main__":
    main()
