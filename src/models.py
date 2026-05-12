from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.features import make_basic_preprocessor


def make_baseline_model() -> Pipeline:
    estimator = ExtraTreesRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=1,
        min_samples_leaf=2,
    )
    return Pipeline(
        steps=[
            ("preprocess", make_basic_preprocessor()),
            ("model", MultiOutputRegressor(estimator, n_jobs=1)),
        ]
    )
