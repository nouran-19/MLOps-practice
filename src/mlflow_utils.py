"""MLflow + DagsHub experiment tracking utilities.

Centralises all MLflow interaction so the training pipeline stays
tracker-agnostic.  Swap this module to switch from MLflow/DagsHub
to W&B, Neptune, or any other backend.
"""

from __future__ import annotations

import os
from typing import Any

import dagshub
import mlflow
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_dagshub_mlflow(
    repo_owner: str = "nouran-19",
    repo_name: str = "MLOps-practice",
) -> None:
    """Bootstrap MLflow tracking through DagsHub.

    ``dagshub.init()`` sets ``MLFLOW_TRACKING_URI`` and injects
    credentials from the ``DAGSHUB_USER_TOKEN`` env-var so that
    ``mlflow.*`` calls transparently hit the hosted DagsHub server.
    No self-hosted MLflow infrastructure is required.

    If ``DAGSHUB_USER_TOKEN`` is not set, MLflow falls back to
    local file-based tracking (``./mlruns``) so development can
    continue offline.
    """
    token = os.environ.get("DAGSHUB_USER_TOKEN", "")

    if not token:
        import warnings
        warnings.warn(
            "DAGSHUB_USER_TOKEN not set — MLflow will log runs locally "
            "to ./mlruns instead of DagsHub. Set the token in .env to "
            "enable remote experiment tracking.",
            stacklevel=2,
        )
        return

    # The DagsHub SDK caches os.environ["DAGSHUB_USER_TOKEN"] at
    # *import time* (in dagshub.common.config).  Since our imports
    # happen before load_dotenv() runs, the cached value is None.
    # Patch it explicitly so dagshub.init() finds the token and
    # skips the interactive OAuth flow (which also crashes on
    # Windows cp1252 consoles).
    import dagshub.common.config as _dagshub_config
    _dagshub_config.token = token
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)


# ---------------------------------------------------------------------------
# Param / metric helpers
# ---------------------------------------------------------------------------

def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dict into dot-notation keys."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_hydra_params(cfg: DictConfig) -> None:
    """Log every Hydra config leaf as an MLflow parameter.

    Parameters are flattened into dot-notation (e.g.
    ``training.test_size``, ``models.random_forest.n_estimators``).
    This means every single hyperparameter is captured in the MLflow
    UI — making any run fully reproducible from the dashboard alone.
    """
    params = _flatten_dict(OmegaConf.to_container(cfg, resolve=True))
    # MLflow has a 500-param limit per batch; chunk if needed.
    param_items = list(params.items())
    batch_size = 100
    for i in range(0, len(param_items), batch_size):
        batch = dict(param_items[i : i + batch_size])
        # Convert list/complex values to strings for MLflow compatibility
        sanitised = {k: str(v) if isinstance(v, (list, dict)) else v for k, v in batch.items()}
        mlflow.log_params(sanitised)


def log_cv_metrics(cv_scores: dict[str, dict[str, float]]) -> None:
    """Log per-candidate cross-validation scores.

    Each model candidate gets its metrics prefixed with the model name
    (e.g. ``random_forest.cv.accuracy``).
    """
    for model_name, metrics in cv_scores.items():
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"{model_name}.cv.{metric_name}", value)


def log_validation_metrics(metrics: dict[str, float], prefix: str = "val") -> None:
    """Log final hold-out validation metrics."""
    for name, value in metrics.items():
        mlflow.log_metric(f"{prefix}.{name}", value)


def log_model_artifact(model_path: str) -> None:
    """Attach the trained model file to the current MLflow run.

    The model is logged as a *generic artifact* (not via
    ``mlflow.sklearn.log_model``) because the pipeline already
    serialises with ``joblib``.  This keeps DVC as the primary
    heavyweight binary store while MLflow gets a lightweight
    run-attached snapshot for traceability.
    """
    mlflow.log_artifact(model_path)


def log_report_artifact(report_path: str) -> None:
    """Attach the JSON training report to the current run."""
    mlflow.log_artifact(report_path)
