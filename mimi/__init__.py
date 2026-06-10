"""
mimi — Keep your models balanced.

Continuous fine-tuning with automatic forgetting detection and skill rollback.

Quick start::

    from mimi import Model

    model = Model("meta-llama/Llama-3.2-1B", strategy="lora")
    model.snapshot(name="before_v1")
    model.learn("data.jsonl", epochs=3)
    report = model.check()
    if not report.is_healthy:
        model.rollback(to="before_v1")
"""

from .detector import ForgettingDetector, ForgettingReport, CategoryComparison
from .live import LiveLearner
from .model import Model, KeelError
from .rollback import RollbackManager
from .snapshot import SkillScore, SkillSnapshot

__all__ = [
    "Model",
    "KeelError",
    "SkillSnapshot",
    "SkillScore",
    "ForgettingDetector",
    "ForgettingReport",
    "CategoryComparison",
    "RollbackManager",
    "LiveLearner",
]

__version__ = "0.1.0"
