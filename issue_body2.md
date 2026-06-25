### Describe the bug

When users try to use QLoRA strategy (`--strategy qlora`), they get a cryptic `ImportError` because `bitsandbytes` is not listed in the optional dependencies.

### Current State

The `pyproject.toml` has optional dependencies for `wandb`, `mlflow`, `neptune`, `serve`, `dev`, but not for `qlora`/`bitsandbytes`.

### Expected Behavior

Users should be able to install QLoRA support with:
```bash
pip install pyrecall[qlora]
```

### Fix

Add `bitsandbytes` to optional dependencies in `pyproject.toml`:
```toml
[project.optional-dependencies]
qlora = ["bitsandbytes>=0.43.0"]
# or combine with cuda-specific extras
```

### Acceptance Criteria

- [ ] `pip install pyrecall[qlora]` installs bitsandbytes
- [ ] QLoRA works after installing with the extra
- [ ] Error message for missing bitsandbytes is clear and suggests the extra