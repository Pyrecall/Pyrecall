# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.12.x  | ✅ |
| 0.11.x  | ✅ security fixes only |
| < 0.11  | ❌ |

Only the two most recent minor releases receive security fixes. Users on older versions are encouraged to upgrade.

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.** Public disclosure before a fix is available puts all users at risk.

Instead, report vulnerabilities privately via one of the following:

- **GitHub private vulnerability reporting** (preferred): [github.com/Pyrecall/Pyrecall/security/advisories/new](https://github.com/Pyrecall/Pyrecall/security/advisories/new)
- **Email**: [awaneesh.ranjan@gmail.com](mailto:awaneesh.ranjan@gmail.com) — use the subject line `[pyrecall] Security Vulnerability`

### What to include

- A clear description of the vulnerability and its potential impact
- The affected version(s)
- Steps to reproduce or a minimal proof-of-concept
- Any suggested fixes or mitigations, if you have them

### Response timeline

| Milestone | Target |
| --------- | ------ |
| Acknowledgement | within 48 hours |
| Initial assessment | within 5 business days |
| Fix or mitigation | within 30 days for critical/high severity; 90 days for medium/low |
| Public disclosure | coordinated with reporter after fix is released |

We follow [coordinated disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure). If you would like to be credited in the release notes, please let us know in your report.

## Scope

Vulnerabilities we consider in scope:

- **Path traversal** — snapshot names or file paths reaching the filesystem unsanitized
- **XSS** — user-controlled data rendered unescaped in HTML report output
- **Arbitrary code execution** — via malicious model weights, JSONL training data, or config files
- **Credential exposure** — HuggingFace tokens, Neptune/W&B API keys, or passphrases logged or stored insecurely
- **Unsafe deserialization** — loading untrusted snapshot files leading to code execution
- **Dependency vulnerabilities** — critical CVEs in `transformers`, `peft`, `torch`, or other direct dependencies

Out of scope:

- Vulnerabilities in models downloaded from HuggingFace Hub (report those upstream)
- Issues that require physical access to the machine running pyrecall
- Social engineering attacks
- Denial-of-service via extremely large training files (resource exhaustion is expected with user-supplied data)

## Dependency security

pyrecall pins its direct dependencies in `pyproject.toml`. We monitor for known CVEs via GitHub Dependabot. If you discover a vulnerable transitive dependency before it is flagged automatically, please report it using the process above.

## Security-relevant design decisions

- Snapshot names are validated against path traversal patterns (`..`, `/`, `\`) before any filesystem operation — see `rollback.py:_validate_snapshot_name()`
- Snapshot names are HTML-escaped before inclusion in report output — see `detector.py:to_html()`
- Encryption passphrases are never stored on disk; only a PBKDF2-derived salt is persisted — see `encrypt.py`
- `--dry-run` snapshots never write adapter weights or update the baseline config
