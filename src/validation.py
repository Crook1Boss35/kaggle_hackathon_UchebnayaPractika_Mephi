import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

from src.config import RANDOM_STATE, TARGET_COLUMNS
from src.metrics import competition_score


def cross_validate_model(model, X: pd.DataFrame, y: pd.DataFrame, n_splits: int = 5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((len(X), len(TARGET_COLUMNS)))
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        valid_pred = fold_model.predict(X_valid)
        valid_pred = np.clip(valid_pred, a_min=0.0, a_max=None)
        oof[valid_idx] = valid_pred

        scores = competition_score(y_valid, valid_pred)
        scores["fold"] = fold
        fold_scores.append(scores)

    overall_scores = competition_score(y, oof)
    return oof, pd.DataFrame(fold_scores), overall_scores

