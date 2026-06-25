### Feature Request

The `pyrecall diff` command has `--output` / `-o` flag to save reports, but the help text mentions `--save-report` as an alias that doesn't exist.

### Current State

In `cli.py`, the `diff` command has:
```python
output: Annotated[
    str | None,
    typer.Option(
        "--output",
        "--save-report",  # This alias is defined but may not work
        "-o",
        help="Save the report to a file. Format inferred from extension: .html, .md, or .json.",
    ),
] = None,
```

### Issue

The `--save-report` alias is defined in the `Option` but may not be functional. Need to verify and ensure both `--output` and `--save-report` work.

### Acceptance Criteria

- [ ] `pyrecall diff before after --save-report report.html` works
- [ ] `pyrecall diff before after -o report.html` works
- [ ] Help text shows both aliases