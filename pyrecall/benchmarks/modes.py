"""Benchmark mode configuration for configurable evaluation sizes."""

from __future__ import annotations

from enum import Enum


class BenchmarkMode(Enum):
    """Benchmark execution modes with different prompt counts."""

    FAST = "fast"
    STANDARD = "standard"
    FULL = "full"

    @property
    def prompt_count(self) -> int:
        """Return the approximate number of prompts for this mode."""
        return {
            BenchmarkMode.FAST: 30,
            BenchmarkMode.STANDARD: 100,
            BenchmarkMode.FULL: 180,
        }[self]

    @property
    def description(self) -> str:
        """Return a human-readable description of this mode."""
        return {
            BenchmarkMode.FAST: "Quick validation during development",
            BenchmarkMode.STANDARD: "Balanced evaluation for regular workflows",
            BenchmarkMode.FULL: "Comprehensive benchmark before deployment or release",
        }[self]


# Default mode to use when not specified
DEFAULT_MODE = BenchmarkMode.STANDARD
