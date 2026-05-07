import hydra
from omegaconf import DictConfig, OmegaConf

from src.logger import ExecutorLogger
from src.pipeline import train_titanic_pipeline

logger = ExecutorLogger("training")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Training started")
    logger.info("Resolved config:\n{}", OmegaConf.to_yaml(cfg, resolve=True))
    train_titanic_pipeline(cfg.pipeline, logger)
    logger.info("Training finished")


if __name__ == "__main__":
    main()
