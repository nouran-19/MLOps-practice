from functools import partial
import os
import pickle
from typing import Any, Dict

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from src.fake.estimator import FakeEstimator

SOURCE = os.path.join("data", "processed")
MODEL_PATH = "models"
N_FOLDS = 5
MAX_EVALS = 4
SPACE = {
    "random_state": scope.int(hp.quniform("random_state", 2, 80, 1)),
}


def encode_target_col(
    file_name: str,
    target_col: str,
    model_name: str,
    logger,
):
    train_df = pd.read_parquet(os.path.join(SOURCE, f"{file_name}-train.parquet"))
    test_df = pd.read_parquet(os.path.join(SOURCE, f"{file_name}-test.parquet"))
    X_train, y_train = train_df.drop(target_col, axis=1), train_df[target_col]
    X_test, y_test = test_df.drop(target_col, axis=1), test_df[target_col]
    logger.info("Fitting the encoder/decoder of target variable")
    logger.info(f"Number of classes: {len(y_train.unique())}")
    encoder = {class_: idx for idx, class_ in enumerate(y_train.unique())}
    decoder = {idx: class_ for class_, idx in encoder.items()}
    # save the encoder/decoder of target
    label_translator = {"encoder": encoder, "decoder": decoder}
    logger.info("encoder/decoder of target created successfully")
    if not os.path.exists(os.path.join(MODEL_PATH, model_name)):
        os.makedirs(os.path.join(MODEL_PATH, model_name))
    with open(
        os.path.join(MODEL_PATH, model_name, "model_target_translator.pkl"),
        "wb",
    ) as pkl:
        pickle.dump(label_translator, pkl)
    logger.info("encoder/decoder of target saved")
    return X_train, y_train, X_test, y_test


def objective(params: Dict[str, Any], X, y, n_folds: int = N_FOLDS) -> Dict[str, Any]:
    model = FakeEstimator(**params)
    scores = cross_validate(model, X, y, cv=n_folds, n_jobs=-1, scoring="accuracy")
    score = np.mean(scores["test_score"])
    return {"loss": score, "params": params, "status": STATUS_OK}


def trainer(X, y, model_name: str, logger) -> None:
    logger.info("Load encoder/decoder of target variable")
    with open(
        os.path.join(MODEL_PATH, model_name, "model_target_translator.pkl"),
        "rb",
    ) as pkl:
        translator = pickle.load(pkl)
    y_train_enc = y.apply(lambda x: translator["encoder"][x])
    bayes_trials = Trials()
    fmin_objective = partial(objective, X=X, y=y_train_enc)
    logger.info("optimization started")
    fmin(
        fn=fmin_objective,
        space=SPACE,
        algo=tpe.suggest,
        max_evals=MAX_EVALS,
        trials=bayes_trials,
    )
    logger.info("optimization completed")
    best_model = bayes_trials.results[
        np.argmin([r["loss"] for r in bayes_trials.results])
    ]
    params = best_model["params"]
    final_model = FakeEstimator(**params)
    final_model.fit(X, y_train_enc)
    logger.info("save the final optimized model")
    if not os.path.exists(os.path.join(MODEL_PATH, model_name)):
        os.makedirs(os.path.join(MODEL_PATH, model_name))
    with open(os.path.join(MODEL_PATH, model_name, "final_model.pkl"), "wb") as pkl:
        pickle.dump(final_model, pkl)
    logger.info("model trained and saved successfully")