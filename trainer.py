import sys
from pathlib import Path

from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

# MLflow prints emoji (🏃) when closing runs, which crashes on
# Windows legacy consoles using cp1252 encoding. Force UTF-8.
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.logger import ExecutorLogger
from src.pipeline import train_titanic_pipeline

# Load .env before Hydra changes cwd — ensures DAGSHUB_USER_TOKEN
# and other secrets are available regardless of the Hydra output dir.
load_dotenv(Path(__file__).resolve().parent / ".env")

logger = ExecutorLogger("training")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Training started")
    logger.info("Resolved config:\n{}", OmegaConf.to_yaml(cfg, resolve=True))
    train_titanic_pipeline(cfg.pipeline, logger)
    logger.info("Training finished")


if __name__ == "__main__":
    main()
