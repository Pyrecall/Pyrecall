"""
Pyrecall — Keep your models balanced.

Continuous fine-tuning with automatic forgetting detection and skill rollback.

Quick start:

    from pyrecall import Model

    model = Model("meta-llama/Llama-3.2-1B", strategy="lora")
    model.snapshot(name="before_v1")
    model.learn("data.jsonl", epochs=3)
    report = model.check()
    if not report.is_healthy:
        model.rollback(to="before_v1")
"""

from importlib.metadata import PackageNotFoundError, version

from .detector import CategoryComparison, ForgettingDetector, ForgettingReport
from .live import LiveLearner
from .model import Model, PyrecallError
from .replay import ReplayBuffer
from .rollback import RollbackManager
from .snapshot import SkillScore, SkillSnapshot
from .trackers import MLflowTracker, NeptuneTracker, SnapshotTracker, WandbTracker

__all__ = [
    "Model",
    "PyrecallError",
    "SkillSnapshot",
    "SkillScore",
    "ForgettingDetector",
    "ForgettingReport",
    "CategoryComparison",
    "RollbackManager",
    "LiveLearner",
    "ReplayBuffer",
    "SnapshotTracker",
    "WandbTracker",
    "MLflowTracker",
    "NeptuneTracker",
]

try:
    __version__ = version("pyrecall")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"
