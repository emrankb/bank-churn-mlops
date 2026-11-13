# Bank churn MLOps

Incremental capstone for preparing customer churn data, comparing predictive
models, tracking experiments, and serving a churn prediction API.

## Current status

- **Task 1:** Data validation, descriptive analysis, and leakage-safe preprocessing
- **Task 2:** Model training, cross-validated F1 comparison, held-out metrics,
  and local model artifacts
- **Task 3:** Local MLflow experiment tracking, F1-selected model registration,
  registry versioning, and a `champion` alias
- **Task 4:** FastAPI, Docker, monitoring, and retraining remain for a later milestone

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

## MLflow experiment tracking

MLflow logs the three classifier candidates, their training-fold metrics and
parameters, and the held-out metrics of the selected model. The selected
pipeline is registered locally as `bank-churn-classifier` with a version and
`champion` alias. The SQLite tracking database and model artifacts are
private local files, never committed.

```bash
python -m pip install -r requirements-dev.txt -r requirements-mlflow.txt
python -m scripts.track_experiments
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1 --port 5000
```

Browse `http://127.0.0.1:5000` in the local environment for side-by-side
experiment runs. This command trains the model and makes no external model
API calls. See `docs/mlflow_tracking.md` for registry versions, privacy
notes, and the local database.

## Tests

```bash
python -m pytest -q
```

GitHub Actions runs synthetic-data tests without the private customer CSV or
paid services. CI pass counts are software tests, **not real-data prediction
metrics**. The final churn-model scores require a local run against the
original dataset. CI tests MLflow integration against invented customer rows. No MLflow scores
from the private original dataset have been claimed yet.

See `docs/data_preparation.md` and `docs/model_training.md` for details.
