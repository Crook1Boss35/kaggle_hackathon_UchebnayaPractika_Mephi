import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from src.config import OUTPUT_DIR, TARGET_COLUMNS
from src.data import load_train_test, split_features_targets
from src.models import make_baseline_model
from src.submission import make_submission
from src.validation import cross_validate_model


def main() -> None:
    train, test, sample_submission = load_train_test()
    X, y, X_test = split_features_targets(train, test)

    model = make_baseline_model()
    oof, fold_scores, overall_scores = cross_validate_model(model, X, y)

    model.fit(X, y)
    test_predictions = model.predict(X_test)
    submission = make_submission(sample_submission, test_predictions)

    (OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "oof").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "submissions").mkdir(parents=True, exist_ok=True)

    report = fold_scores.copy()
    report.loc[len(report)] = {
        **{target: overall_scores[target] for target in TARGET_COLUMNS},
        "score": overall_scores["score"],
        "fold": "overall",
    }
    report.to_csv(OUTPUT_DIR / "reports" / "baseline_report.csv", index=False)

    oof_df = pd.DataFrame(oof, columns=TARGET_COLUMNS)
    oof_df.insert(0, "index", train["index"])
    oof_df.to_csv(OUTPUT_DIR / "oof" / "baseline_oof.csv", index=False)

    submission.to_csv(OUTPUT_DIR / "submissions" / "baseline_submission.csv", index=False)

    print("Baseline CV scores:")
    print(report.to_string(index=False))
    print()
    print(f"Submission saved to {OUTPUT_DIR / 'submissions' / 'baseline_submission.csv'}")


if __name__ == "__main__":
    main()
