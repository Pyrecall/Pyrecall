# Changelog

All notable changes to this project are documented in this file, derived from
release history. Versions follow [Semantic Versioning](https://semver.org/).

## [0.12.1] and later

See [GitHub Releases](https://github.com/Pyrecall/Pyrecall/releases) and
`git log` for changes from v0.12.0 onward; this section will be kept current
going forward.

## [0.11.2] - bug fixes and doc corrections

## [0.11.1] - patch fixes

## [0.10.1] - patch fixes

## [0.10.0] - log-likelihood scoring

- Switched benchmark scoring to log-likelihood.
- Added Cohen's d effect size to forgetting detection.
- Expanded default benchmark suite to 160 prompts.

## [0.8.0] - custom benchmark suites

- Added support for user-defined benchmark suites.

## [0.7.0] - compare command

- Added `pyrecall compare`.

## [0.6.1] - Neptune tracker

- Added Neptune experiment tracker integration.

## [0.6.0] - diff command

- Added `pyrecall diff`.

## [0.5.1] - fixes

- Snapshot tests, fixed `cryptography` dependency, README fixes.

## [0.5.0] - experiment trackers

- Added Weights & Biases and MLflow experiment tracker integrations.

## [0.4.0] - benchmark categories

- Added `tool_use` and `advanced_math` benchmark categories.

## [0.3.0] - per-prompt breakdown

- `pyrecall check` now reports a per-prompt breakdown.

## [0.2.1] - live CLI subcommands

- Added live CLI subcommands.
- Added `--no-update-baseline` flag.

## [0.2.0] - snapshot-before

- Added `--snapshot-before` support, `init` validation, rollback smoke test.

## [0.1.5] - replay subcommands

- Added `pyrecall replay status` / `pyrecall replay clear`.

## [0.1.4] - learn command

- Added `pyrecall learn` CLI command.

## [0.1.3] - replay buffer

- Added replay buffer for continual learning.

## [0.1.2] - smoke tests

- Added end-to-end smoke tests.
- Fixed MPS fp16 handling.

## [0.1.1] - input validation

- Added input validation and error handling.

## [0.1.0] - initial release
