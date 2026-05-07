from src.logger import ExecutorLogger
from src.pipeline import train_titanic_pipeline


def main(logger) -> None:
    logger.info("Training started")
    train_titanic_pipeline(logger)
    logger.info("Training finished")


if __name__ == "__main__":
    logger = ExecutorLogger("training")
    main(logger)
