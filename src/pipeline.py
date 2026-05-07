"""Titanic training pipeline with Hydra-configurable settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import zipfile

from hydra.utils import instantiate
import joblib
from kaggle.api.kaggle_api_extended import KaggleApi
from omegaconf import DictConfig
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TITLE_REPLACEMENTS = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Rare",
    "Countess": "Rare",
    "Capt": "Rare",
    "Col": "Rare",
    "Don": "Rare",
    "Dr": "Rare",
    "Major": "Rare",
    "Rev": "Rare",
    "Sir": "Rare",
    "Jonkheer": "Rare",
    "Dona": "Rare",
}


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add Titanic-specific features before preprocessing."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TitanicFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        frame["FamilySize"] = frame["SibSp"].fillna(0) + frame["Parch"].fillna(0) + 1
        frame["IsAlone"] = (frame["FamilySize"] == 1).astype(int)
        frame["HasCabin"] = frame["Cabin"].notna().astype(int)
        frame["Title"] = self._extract_title(frame["Name"])
        frame["Deck"] = (
            frame["Cabin"].fillna("Unknown").astype(str).str[0].replace({"U": "Unknown"})
        )
        frame["TicketPrefix"] = self._extract_ticket_prefix(frame["Ticket"])
        return frame

    @staticmethod
    def _extract_title(names: pd.Series) -> pd.Series:
        titles = names.fillna("").str.extract(r",\s*([^\.]+)\.", expand=False).fillna("Unknown")
        return titles.str.strip().replace(TITLE_REPLACEMENTS)

    @staticmethod
    def _extract_ticket_prefix(tickets: pd.Series) -> pd.Series:
        prefixes = tickets.fillna("").astype(str)
        prefixes = prefixes.str.replace(r"\d+", "", regex=True)
        prefixes = prefixes.str.replace(r"[^A-Za-z]", "", regex=True).str.upper()
        return prefixes.replace("", "NONE")


def _build_preprocessor(cfg: DictConfig) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, list(cfg.preprocessing.numeric_features)),
            ("categorical", categorical_transformer, list(cfg.preprocessing.categorical_features)),
        ]
    )


def _build_model_candidates(cfg: DictConfig) -> dict[str, Any]:
    return {model_name: instantiate(model_cfg) for model_name, model_cfg in cfg.models.items()}


def _build_pipeline(cfg: DictConfig, estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineering", TitanicFeatureEngineer()),
            ("preprocessing", _build_preprocessor(cfg)),
            ("model", estimator),
        ]
    )


def download_titanic_competition_data(cfg: DictConfig, logger) -> None:
    """Download and extract the Titanic competition data from Kaggle."""

    raw_data_dir = Path(cfg.data.raw_data_dir)
    train_file = cfg.data.train_file
    test_file = cfg.data.test_file
    competition_name = cfg.data.competition_name

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    train_path = raw_data_dir / train_file
    test_path = raw_data_dir / test_file

    if train_path.exists() and test_path.exists():
        logger.info("Titanic data already available locally")
        return

    logger.info("Downloading Titanic competition data from Kaggle")
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(competition_name, path=str(raw_data_dir), quiet=False)

    archive_path = raw_data_dir / f"{competition_name}.zip"
    if archive_path.exists():
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(raw_data_dir)
        archive_path.unlink()

    if not train_path.exists():
        raise FileNotFoundError(
            "Titanic train.csv was not found after download. Check Kaggle authentication and "
            "competition access."
        )

    logger.info("Titanic competition data is ready")


def load_titanic_data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    train_data_path = Path(cfg.data.raw_data_dir) / cfg.data.train_file
    data = pd.read_csv(train_data_path)
    target_column = cfg.data.target_column
    drop_columns = list(cfg.data.drop_columns)

    if target_column not in data.columns:
        raise ValueError(f"Titanic training data must contain target column: {target_column}")

    y = data[target_column]
    X = data.drop(columns=[target_column])
    present_drop_columns = [column for column in drop_columns if column in X.columns]
    if present_drop_columns:
        X = X.drop(columns=present_drop_columns)
    return X, y


def evaluate_candidates(
    cfg: DictConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    logger,
) -> tuple[str, Pipeline, dict[str, dict[str, float]]]:
    cv = StratifiedKFold(
        n_splits=cfg.training.cv_splits,
        shuffle=True,
        random_state=cfg.training.random_state,
    )
    scoring_metrics = list(cfg.training.scoring)
    selection_metric = cfg.training.selection_metric
    scoring = {metric: metric for metric in scoring_metrics}
    if selection_metric not in scoring:
        raise ValueError(
            f"selection_metric={selection_metric} must be included in training.scoring"
        )

    candidates = _build_model_candidates(cfg)
    scores: dict[str, dict[str, float]] = {}
    best_name = ""
    best_score = float("-inf")
    best_pipeline: Pipeline | None = None

    for model_name, estimator in candidates.items():
        pipeline = _build_pipeline(cfg, estimator)
        cv_result = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            n_jobs=-1,
            scoring=scoring,
        )
        model_scores = {
            metric: float(cv_result[f"test_{metric}"].mean()) for metric in scoring_metrics
        }
        scores[model_name] = model_scores
        score_message = ", ".join(
            f"{metric}={model_scores[metric]:.4f}" for metric in scoring_metrics
        )
        logger.info(f"{model_name} cross-validation scores: {score_message}")
        if model_scores[selection_metric] > best_score:
            best_score = model_scores[selection_metric]
            best_name = model_name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No Titanic model candidate was selected")

    logger.info(f"Selected best model candidate: {best_name}")
    return best_name, best_pipeline, scores


def train_titanic_pipeline(cfg: DictConfig, logger) -> None:
    download_titanic_competition_data(cfg, logger)
    X, y = load_titanic_data(cfg)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
        stratify=y,
    )

    best_name, best_pipeline, cv_scores = evaluate_candidates(cfg, X_train, y_train, logger)

    logger.info("Fitting the best pipeline on the training split")
    best_pipeline.fit(X_train, y_train)

    valid_predictions = best_pipeline.predict(X_valid)
    valid_probabilities = best_pipeline.predict_proba(X_valid)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_valid, valid_predictions)),
        "precision": float(precision_score(y_valid, valid_predictions)),
        "recall": float(recall_score(y_valid, valid_predictions)),
        "f1": float(f1_score(y_valid, valid_predictions)),
        "roc_auc": float(roc_auc_score(y_valid, valid_probabilities)),
    }

    model_dir = Path(cfg.artifacts.model_dir)
    report_dir = Path(cfg.artifacts.report_dir)
    model_path = model_dir / cfg.artifacts.model_file
    report_path = report_dir / cfg.artifacts.report_file

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)

    report = {
        "dataset": f"kaggle/{cfg.data.competition_name}",
        "config_name": cfg.experiment_name,
        "best_model": best_name,
        "cross_validation": cv_scores,
        "validation_metrics": metrics,
        "artifact_path": str(model_path),
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4)

    logger.info(f"Saved trained pipeline to {model_path}")
    logger.info(f"Saved training report to {report_path}")
