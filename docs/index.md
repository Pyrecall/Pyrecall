# Pyrecall

Forgetting detection and skill rollback for fine-tuned LLMs.

This site renders the docstrings already written throughout the `pyrecall`
package into browsable API reference pages. Build it locally with:

```bash
pip install -e ".[docs]"
mkdocs serve
```

See the [README](https://github.com/Pyrecall/Pyrecall#readme) for
installation and a quickstart, and
[examples/basic_workflow.py](https://github.com/Pyrecall/Pyrecall/blob/main/examples/basic_workflow.py)
for a complete end-to-end script.

## API Reference

- [`Model`](api/model.md) — snapshot, learn, check, diff, and rollback
- [`Detector`](api/detector.md) — forgetting detection and scoring
- [`Rollback`](api/rollback.md) — snapshot save/load/restore
- [`Snapshot`](api/snapshot.md) — snapshot data structures
- [`Trackers`](api/trackers.md) — W&B / MLflow / Neptune integrations
