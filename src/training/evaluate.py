import argparse
import json
import os
from pathlib import Path
import pickle

import joblib
import pandas as pd
from skore import EstimatorReport
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.logger import ExecutorLogger

MODEL_PATH = "models"
REPORT_PATH = "reports"


def evaluate(X_test, y_test, model_name: str, logger) -> None:
    logger.info("loading model")
    with open(os.path.join(MODEL_PATH, model_name, "final_model.pkl"), "rb") as pkl:
        final_model = pickle.load(pkl)
    with open(
        os.path.join(MODEL_PATH, model_name, "model_target_translator.pkl"),
        "rb",
    ) as pkl:
        translator = pickle.load(pkl)
    y_test_enc = y_test.apply(lambda x: translator["encoder"][x])
    final_report = EstimatorReport(final_model, X_test=X_test, y_test=y_test_enc)
    logger.info("creating evaluation report")
    evaluation_report = {
        "model_name": model_name,
        "estimator_name": final_report.estimator_name_,
        "fitting_time": final_report.fit_time_,
        "accuracy": final_report.metrics.accuracy(),
        "precision": final_report.metrics.precision(),
        "recall": final_report.metrics.recall(),
        "prediction_time": final_report.metrics.timings(),
    }
    logger.info("saving evaluation report")
    if not os.path.exists(os.path.join(REPORT_PATH, model_name)):
        os.makedirs(os.path.join(REPORT_PATH, model_name))
    with open(
        os.path.join(REPORT_PATH, model_name, "evaluation_report.json"), "w"
    ) as js:
        json.dump(evaluation_report, js, indent=4)


def evaluate_titanic_pipeline(
    model_path: str,
    data_path: str,
    predictions_path: str,
    report_path: str,
    target_column: str = "Survived",
    drop_columns: list[str] | None = None,
) -> None:
    """Evaluate a trained Titanic pipeline model on a CSV file."""
    logger = ExecutorLogger("evaluation")

    logger.info("Loading trained model from {}", model_path)
    model = joblib.load(model_path)

    frame = pd.read_csv(data_path)
    has_target = target_column in frame.columns

    X = frame.drop(columns=[target_column], errors="ignore")
    if drop_columns:
        removable = [column for column in drop_columns if column in X.columns]
        X = X.drop(columns=removable)

    predictions = model.predict(X)

    predictions_output = frame.copy()
    predictions_output["prediction"] = predictions
    predictions_file = Path(predictions_path)
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    predictions_output.to_csv(predictions_file, index=False)

    report: dict[str, object] = {
        "model_path": model_path,
        "data_path": data_path,
        "rows_scored": int(len(frame)),
        "predictions_path": str(predictions_file),
    }

    if has_target:
        y_true = frame[target_column]
        report["metrics"] = {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
        }

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4)

    logger.info("Saved downstream predictions to {}", predictions_file)
    logger.info("Saved downstream report to {}", report_file)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Titanic model as downstream DVC stage")
    parser.add_argument(
        "--model-path",
        default="models/titanic/titanic_pipeline.pkl",
        help="Path to trained model pipeline artifact",
    )
    parser.add_argument(
        "--data-path",
        default="data/raw/titanic/train.csv",
        help="Path to CSV file to score",
    )
    parser.add_argument(
        "--predictions-path",
        default="reports/titanic/downstream_predictions.csv",
        help="Path for scored output CSV",
    )
    parser.add_argument(
        "--report-path",
        default="reports/titanic/downstream_report.json",
        help="Path for downstream metrics report",
    )
    parser.add_argument(
        "--target-column",
        default="Survived",
        help="Optional target column used for metrics if present",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=["PassengerId"],
        help="Columns to drop before inference",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_titanic_pipeline(
        model_path=args.model_path,
        data_path=args.data_path,
        predictions_path=args.predictions_path,
        report_path=args.report_path,
        target_column=args.target_column,
        drop_columns=args.drop_columns,
    )