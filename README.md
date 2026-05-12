# Influenza Molecule Activity Prediction

This repository contains a reproducible baseline for predicting three biological activity targets for molecular compounds:

- `IC50`: concentration that suppresses 50% of viral activity
- `CC50`: concentration toxic for 50% of cells
- `SI`: selectivity index

The input data contains numeric molecular descriptors generated from chemical structures.

## Repository Structure

```text
.
├── configs/                 # Model and experiment configs
├── data/                    # Local competition data location
├── notebooks/               # EDA and experiment notebooks
├── outputs/                 # Generated OOF predictions, reports and submissions
├── scripts/                 # CLI entrypoints
├── src/                     # Reusable training code
└── presentation/            # Final project slides
```

## Data

Place the competition files in the project root or in `data/raw/`:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

Current local files:

- `train.csv`: 751 rows, 210 feature columns, 3 targets
- `test.csv`: 250 rows, 210 feature columns
- `sample_submission.csv`: expected submission format

## Approach

Day 1 baseline:

1. Load train/test data.
2. Split features and targets.
3. Remove constant features.
4. Impute missing numeric values with median.
5. Train a simple multi-target baseline using fixed cross-validation.
6. Generate a valid submission file.

The baseline intentionally stays simple so all later model experiments can reuse the same validation, preprocessing and submission format.

## Reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the quick baseline:

```bash
python scripts/train_baseline.py
```

The script writes:

- `outputs/reports/baseline_report.csv`
- `outputs/oof/baseline_oof.csv`
- `outputs/submissions/baseline_submission.csv`

## Next Experiments

Recommended next steps:

- compare raw targets vs `log1p` targets;
- train separate models for `IC50`, `CC50`, and `SI`;
- test `Ridge`, `ElasticNet`, `ExtraTrees`, `RandomForest`, `HistGradientBoosting`, `CatBoost`, `LightGBM`, and `XGBoost`;
- blend the best OOF/test predictions.

