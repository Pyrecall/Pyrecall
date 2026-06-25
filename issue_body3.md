### Feature Request

Add a `--watch` flag to `pyrecall check` that continuously monitors for new snapshots and runs forgetting detection automatically. This already exists in the CLI - need to verify it works and document it.

### Current State

The `pyrecall check --watch` command exists but may need improvements:

1. **Documentation**: README doesn't mention `--watch` for `check`
2. **Exit codes**: Should exit with 2 if last check detected forgetting
3. **Baseline selection**: Should allow specifying which snapshots to compare

### Acceptance Criteria

- [ ] `pyrecall check --watch --interval 30` works as documented
- [ ] Exit code 2 on forgetting, 0 on healthy, 1 on error
- [ ] Works with `--before` and `--after` to compare specific snapshots
- [ ] README updated with `--watch` usage example
- [ ] Help text clearly explains the behavior" --label "enhancement" --milestone "v0.12.1"