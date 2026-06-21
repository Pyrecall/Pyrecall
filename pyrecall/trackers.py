"""Experiment tracker integrations for logging snapshot scores to W&B, MLflow, or Neptune."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .snapshot import SkillSnapshot


@runtime_checkable
class SnapshotTracker(Protocol):
    """Protocol that any tracker must satisfy."""

    def log_snapshot(self, snapshot: SkillSnapshot) -> None: ...


class WandbTracker:
    """
    Log snapshot scores to Weights & Biases.

    Requires ``wandb`` to be installed::

        pip install pyrecall[wandb]

    Each snapshot becomes a W&B run named after the snapshot.  Per-category
    scores are logged as ``pyrecall/<category>`` metrics, plus
    ``pyrecall/overall``.

    Example::

        from pyrecall import Model
        from pyrecall.trackers import WandbTracker

        model = Model("meta-llama/Llama-3.2-1B")
        tracker = WandbTracker(project="my-finetune")
        model.snapshot("before_v1", tracker=tracker)
    """

    def __init__(self, project: str = "pyrecall", **wandb_init_kwargs) -> None:
        self.project = project
        self._init_kwargs = wandb_init_kwargs

    def log_snapshot(self, snapshot: SkillSnapshot) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Install it with: pip install pyrecall[wandb]"
            ) from exc

        metrics: dict[str, float] = {
            f"pyrecall/{cat}": score for cat, score in snapshot.category_scores().items()
        }
        metrics["pyrecall/overall"] = snapshot.overall_score()

        run = wandb.init(
            project=self.project,
            name=snapshot.name,
            reinit=True,
            tags=["pyrecall", snapshot.model_name],
            **self._init_kwargs,
        )
        try:
            run.log(metrics)
        finally:
            run.finish()

    def log_step(self, step: int, loss: float) -> None:
        try:
            import wandb
        except ImportError:
            return
        wandb.log({"train/loss": loss, "train/step": step}, step=step)


class MLflowTracker:
    """
    Log snapshot scores to MLflow.

    Requires ``mlflow`` to be installed::

        pip install pyrecall[mlflow]

    Each snapshot becomes an MLflow run named after the snapshot under
    the configured experiment.  Per-category scores are logged as
    ``pyrecall.<category>`` metrics, plus ``pyrecall.overall``.

    Example::

        from pyrecall import Model
        from pyrecall.trackers import MLflowTracker

        model = Model("meta-llama/Llama-3.2-1B")
        tracker = MLflowTracker(experiment_name="my-finetune")
        model.snapshot("before_v1", tracker=tracker)
    """

    def __init__(
        self,
        experiment_name: str = "pyrecall",
        tracking_uri: str | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

    def log_snapshot(self, snapshot: SkillSnapshot) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                "mlflow is not installed. Install it with: pip install pyrecall[mlflow]"
            ) from exc

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=snapshot.name):
            metrics: dict[str, float] = {
                f"pyrecall.{cat}": score for cat, score in snapshot.category_scores().items()
            }
            metrics["pyrecall.overall"] = snapshot.overall_score()
            mlflow.log_metrics(metrics)
            mlflow.set_tag("pyrecall.snapshot", snapshot.name)
            mlflow.set_tag("pyrecall.model", snapshot.model_name)

    def log_step(self, step: int, loss: float) -> None:
        try:
            import mlflow
        except ImportError:
            return
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.log_metric("train/loss", loss, step=step)


class NeptuneTracker:
    """
    Log snapshot scores to Neptune.

    Requires ``neptune`` to be installed::

        pip install pyrecall[neptune]

    Each snapshot becomes a Neptune run named after the snapshot.  Per-category
    scores are logged as ``pyrecall/<category>`` fields, plus
    ``pyrecall/overall``.

    Example::

        from pyrecall import Model
        from pyrecall.trackers import NeptuneTracker

        model = Model("meta-llama/Llama-3.2-1B")
        tracker = NeptuneTracker(project="workspace/my-project")
        model.snapshot("before_v1", tracker=tracker)
    """

    def __init__(self, project: str, **neptune_init_kwargs) -> None:
        self.project = project
        self._init_kwargs = neptune_init_kwargs

    def log_snapshot(self, snapshot: SkillSnapshot) -> None:
        try:
            import neptune
        except ImportError as exc:
            raise ImportError(
                "neptune is not installed. Install it with: pip install pyrecall[neptune]"
            ) from exc

        run = neptune.init_run(
            project=self.project,
            name=snapshot.name,
            tags=["pyrecall", snapshot.model_name],
            **self._init_kwargs,
        )
        try:
            for cat, score in snapshot.category_scores().items():
                run[f"pyrecall/{cat}"] = score
            run["pyrecall/overall"] = snapshot.overall_score()

            # Log metadata
            run["pyrecall/metadata/model_name"] = snapshot.model_name
            run["pyrecall/metadata/snapshot_name"] = snapshot.name
            run["pyrecall/metadata/timestamp"] = snapshot.created_at.isoformat()
        finally:
            run.stop()

    def log_step(self, step: int, loss: float) -> None:
        try:
            import neptune
        except ImportError:
            return
        run = neptune.init_run(project=self.project, **self._init_kwargs)
        try:
            run["train/loss"].append(loss, step=step)
        finally:
            run.stop()
