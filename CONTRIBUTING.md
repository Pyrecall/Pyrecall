# Contributors

Thank you to everyone who has contributed to Pyrecall.

## Core Contributors

- **Atharv Ranjan** ([@Arths17](https://github.com/Arths17))
- **Siddharth Lakshminarayanan** ([@Sid294](https://github.com/Sid294))

## How to Contribute

Contributions are welcome. Here is how to get started:

### Reporting Bugs

Open an issue at [github.com/Pyrecall/Pyrecall/issues](https://github.com/Pyrecall/Pyrecall/issues) with:
- A clear description of the bug
- Steps to reproduce
- Expected vs. actual behavior
- Python version, PyTorch version, and OS

### Submitting Changes

1. Fork the repository and create a branch from `main`.
2. Install dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Make your changes and add tests where appropriate.
4. Run the test suite:
   ```bash
   pytest
   ```
5. Lint with ruff:
   ```bash
   ruff check pyrecall/
   ```
6. Open a pull request against `main` with a clear description of what changed and why.

### Slow / Integration Tests

End-to-end tests that load real models are marked `slow` and excluded from the default run. To include them:
```bash
pytest -m slow
```

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
