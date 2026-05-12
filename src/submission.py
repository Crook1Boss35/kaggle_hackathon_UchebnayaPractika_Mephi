import numpy as np
import pandas as pd

from src.config import INDEX_COLUMN, SUBMISSION_TARGET_COLUMNS


def make_submission(
    sample_submission: pd.DataFrame, predictions: np.ndarray
) -> pd.DataFrame:
    submission = sample_submission[[INDEX_COLUMN]].copy()
    for idx, column in enumerate(SUBMISSION_TARGET_COLUMNS):
        submission[column] = np.clip(predictions[:, idx], a_min=0.0, a_max=None)
    return submission

