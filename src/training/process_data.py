import os

import pandas as pd
from sklearn.model_selection import train_test_split

SOURCE = os.path.join("data", "raw")
DESTINATION = os.path.join("data", "processed")


def read_process_data(
    file_name: str,
    id_col: str,
    target_col: str,
    logger,
) -> None:
    logger.info("Data Processing started")
    df = pd.read_csv(os.path.join(SOURCE, f"{file_name}.csv"))
    df.set_index(id_col, inplace=True)
    train_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df[target_col]
    )
    train_df.to_parquet(
        os.path.join(DESTINATION, f"{file_name}-train.parquet"), engine="pyarrow"
    )
    test_df.to_parquet(
        os.path.join(DESTINATION, f"{file_name}-test.parquet"), engine="pyarrow"
    )