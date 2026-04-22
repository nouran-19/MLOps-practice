from src.logger import ExecutorLogger
from src.training.download_data import download_iris_data
from src.training.evaluate import evaluate
from src.training.process_data import read_process_data
from src.training.train import encode_target_col, trainer


def main(logger) -> None:
    logger.info("Training started")
    download_iris_data(logger)
    read_process_data("Iris", "Id", "Species", logger)
    X, y, X_test, y_test = encode_target_col("Iris", "Species", "fake", logger)
    trainer(X, y, "fake", logger)
    evaluate(X_test, y_test, "fake", logger)
    logger.info("Training finished")


if __name__ == "__main__":
    logger = ExecutorLogger("training")
    main(logger)