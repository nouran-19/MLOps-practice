import os
import kagglehub
import shutil

RAW_DATA_DIR = os.path.join("data", "raw")


def download_iris_data(logger) -> str:
    logger.info("Downloading Iris dataset from Kaggle...")

    path = kagglehub.dataset_download("uciml/iris")

    files = os.listdir(path)
    csv_files = [f for f in files if f.endswith(".csv")]

    if not csv_files:
        csv_files = files

    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    destination = os.path.join(RAW_DATA_DIR, "Iris.csv")
    source_file = os.path.join(path, csv_files[0])

    shutil.copy(source_file, destination)

    logger.info(f"Iris dataset downloaded to {destination}")
    return destination
