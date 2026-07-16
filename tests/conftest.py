"""Shared test configuration.

Sandboxes HOME for every test so nothing reads from or writes to the real
``~/.pyrecall`` (custom benchmarks, snapshots, replay buffers, run dirs).
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Windows
