"""
pyrecall — Keep your models balanced.

Continuous fine-tuning with automatic forgetting detection and skill rollback.

Quick start::

    from pyrecall import Model

    model = Model("meta-llama/Llama-3.2-1B", strategy="lora")
    model.snapshot(name="before_v1")
    model.learn("data.jsonl", epochs=3)
    report = model.check()
    if not report.is_healthy:
        model.rollback(to="before_v1")
"""

from .detector import ForgettingDetector, ForgettingReport, CategoryComparison
from .live import LiveLearner
from .model import Model, PyrecallError
from .replay import ReplayBuffer
from .rollback import RollbackManager
from .snapshot import SkillScore, SkillSnapshot

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
]

__version__ = "0.1.3"
