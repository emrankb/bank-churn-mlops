# Bank churn MLOps

Incremental capstone for preparing customer churn data, comparing predictive
models, tracking experiments, and serving a churn prediction API.

## Current status

- **Task 1:** Data validation, descriptive analysis, and leakage-safe preprocessing
- **Task 2:** Model training, cross-validated F1 comparison, held-out metrics,
  and local model artifacts
- **Tasks 3–4:** MLflow tracking, model registration, FastAPI, Docker, monitoring,
  and retraining remain for later milestones

## Local setup

Use Python 3.12. The supplied `Churn_Modeling.csv` is not in Git because it
contains customer records. Save it at:

```text
data/raw/Churn_Modeling.csv
```

Install dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Run aggregate exploratory analysis:

```bash
python -m scripts.inspect_data
```

Train and compare models:

```bash
python -m scripts.train_models
```

Training compares a majority-class baseline, logistic regression, and random
forest using 3-fold cross-validation on the training split. The winner is
selected by mean F1-score, refitted, and evaluated once on an untouched 20%
test set. Preprocessing is fitted inside each training fold, and customer
identifiers, surname, and the label are excluded from inputs.

Local outputs are excluded from Git:

- `reports/eda/summary.json` and `reports/eda/churn_by_geography.png`
- `artifacts/churn-model/model.joblib` (saved fitted pipeline)
- `artifacts/churn-model/metrics.json` (aggregate model scores)

Model files saved through joblib should only be loaded from trusted sources.
The format uses pickle under the hood and is not safe for untrusted input.

## Tests

```bash
python -m pytest -q
```

GitHub Actions runs synthetic-data tests without the private customer CSV or
paid services. CI pass counts are software tests, **not real-data prediction
metrics**. The final churn-model scores require a local run against the
original dataset. No MLflow experiment results have been claimed yet.

See `docs/data_preparation.md` and `docs/model_training.md` for details.
