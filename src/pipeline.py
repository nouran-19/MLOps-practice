"""Titanic training pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import zipfile

import joblib
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
COMPETITION_NAME = "titanic"
RAW_DATA_DIR = Path("data") / "raw" / "titanic"
MODEL_DIR = Path("models") / "titanic"
REPORT_DIR = Path("reports") / "titanic"

NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone", "HasCabin"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "Deck", "TicketPrefix"]

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


def _build_preprocessor() -> ColumnTransformer:
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
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def _build_model_candidates() -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            max_depth=None,
            min_samples_leaf=2,
            n_estimators=300,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def _build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineering", TitanicFeatureEngineer()),
            ("preprocessing", _build_preprocessor()),
            ("model", estimator),
        ]
    )


def download_titanic_competition_data(logger) -> Path:
    """Download and extract the Titanic competition data from Kaggle."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_file = RAW_DATA_DIR / "train.csv"
    test_file = RAW_DATA_DIR / "test.csv"
    if train_file.exists() and test_file.exists():
        logger.info("Titanic data already available locally")
        return RAW_DATA_DIR

    logger.info("Downloading Titanic competition data from Kaggle")
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(COMPETITION_NAME, path=str(RAW_DATA_DIR), quiet=False)

    archive_path = RAW_DATA_DIR / f"{COMPETITION_NAME}.zip"
    if archive_path.exists():
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(RAW_DATA_DIR)
        archive_path.unlink()

    if not train_file.exists():
        raise FileNotFoundError(
            "Titanic train.csv was not found after download. Check Kaggle authentication and"
            " competition access."
        )

    logger.info("Titanic competition data is ready")
    return RAW_DATA_DIR


def load_titanic_data() -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(RAW_DATA_DIR / "train.csv")
    if "Survived" not in data.columns:
        raise ValueError("Titanic training data must contain a Survived target column")

    y = data["Survived"]
    X = data.drop(columns=["Survived"])
    if "PassengerId" in X.columns:
        X = X.drop(columns=["PassengerId"])

    return X, y


def evaluate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    logger,
) -> tuple[str, Pipeline, dict[str, dict[str, float]]]:
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    candidates = _build_model_candidates()
    scores: dict[str, dict[str, float]] = {}
    best_name = ""
    best_score = float("-inf")
    best_pipeline: Pipeline | None = None

    for model_name, estimator in candidates.items():
        pipeline = _build_pipeline(estimator)
        cv_result = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            n_jobs=-1,
            scoring={
                "accuracy": "accuracy",
                "f1": "f1",
                "precision": "precision",
                "recall": "recall",
                "roc_auc": "roc_auc",
            },
        )
        model_scores = {
            metric: float(cv_result[f"test_{metric}"].mean())
            for metric in ("accuracy", "f1", "precision", "recall", "roc_auc")
        }
        scores[model_name] = model_scores
        logger.info(
            f"{model_name} cross-validation scores: "
            f"accuracy={model_scores['accuracy']:.4f}, f1={model_scores['f1']:.4f}, "
            f"precision={model_scores['precision']:.4f}, recall={model_scores['recall']:.4f}, "
            f"roc_auc={model_scores['roc_auc']:.4f}"
        )
        if model_scores["accuracy"] > best_score:
            best_score = model_scores["accuracy"]
            best_name = model_name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No Titanic model candidate was selected")

    logger.info(f"Selected best model candidate: {best_name}")
    return best_name, best_pipeline, scores


def train_titanic_pipeline(logger) -> None:
    download_titanic_competition_data(logger)
    X, y = load_titanic_data()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    best_name, best_pipeline, cv_scores = evaluate_candidates(X_train, y_train, logger)

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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "titanic_pipeline.pkl"
    report_path = REPORT_DIR / "training_report.json"

    joblib.dump(best_pipeline, model_path)
    report = {
        "dataset": "kaggle/titanic",
        "best_model": best_name,
        "cross_validation": cv_scores,
        "validation_metrics": metrics,
        "artifact_path": str(model_path),
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4)

    logger.info(f"Saved trained pipeline to {model_path}")
    logger.info(f"Saved training report to {report_path}")


def main() -> None:
    from src.logger import ExecutorLogger

    logger = ExecutorLogger("training")
    train_titanic_pipeline(logger)


if __name__ == "__main__":
    main()
