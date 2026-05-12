import pandas as pd
from sklearn.metrics import root_mean_squared_error

from src.config import TARGET_COLUMNS


def rmse(y_true, y_pred) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def competition_score(y_true: pd.DataFrame, y_pred: np.ndarray) -> dict[str, float]:
    scores = {}
    for idx, target in enumerate(TARGET_COLUMNS):
        scores[target] = rmse(y_true[target], y_pred[:, idx])
    target_scores = [scores[target] for target in TARGET_COLUMNS]
    scores["score"] = sum(target_scores) / len(target_scores)
    return scores
