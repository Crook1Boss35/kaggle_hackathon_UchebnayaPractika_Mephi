from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, INDEX_COLUMN, ROOT_DIR, TARGET_COLUMNS


def _find_data_file(filename: str) -> Path:
    candidates = [ROOT_DIR / filename, DATA_DIR / filename]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {filename}. Put it in the project root or in {DATA_DIR}."
    )


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(_find_data_file("train.csv"))
    test = pd.read_csv(_find_data_file("test.csv"))
    sample_submission = pd.read_csv(_find_data_file("sample_submission.csv"))
    return train, test, sample_submission


def split_features_targets(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_columns = [
        column for column in train.columns if column not in [INDEX_COLUMN, *TARGET_COLUMNS]
    ]
    X = train[feature_columns].copy()
    y = train[TARGET_COLUMNS].copy()
    X_test = test[feature_columns].copy()
    return X, y, X_test

