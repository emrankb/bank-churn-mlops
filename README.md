# Bank churn MLOps

Incremental capstone for preparing customer churn data, comparing predictive
models, tracking experiments, and serving a churn prediction API.

## Current status

- **Task 1:** Data validation, descriptive analysis, and leakage-safe preprocessing
- **Task 2:** Model training, cross-validated F1 comparison, held-out metrics,
  and local model artifacts
- **Task 3:** Local MLflow experiment tracking, F1-selected model registration,
  registry versioning, and a `champion` alias
- **Task 4:** Local FastAPI prediction API, Docker deployment, labeled-batch monitoring,
  and an opt-in scheduled retraining workflow

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

## Prediction API and monitoring

After training a local model using `python -m scripts.train_models`:

```bash
python -m pip install -r requirements-api.txt
python -m scripts.serve_api
```

The FastAPI service listens at `127.0.0.1:8000`. Visit `/docs` for the
prediction input schema and `/health` for readiness. `POST /predict`
accepts the ten model features and returns a binary churn prediction and
estimated churn probability. Requests with customer IDs, names or outcome
labels are rejected. The service does not log customer request bodies.

To build the Docker image and serve the locally trained model without
including it in the image:

```bash
docker build -t bank-churn-mlops .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v "$(pwd)/artifacts/churn-model:/app/artifacts/churn-model:ro" \
  bank-churn-mlops
```

Monitoring uses a separate recent CSV containing *known* churn outcomes.
It compares observed accuracy with the saved test baseline and computes
simple feature-distribution shift indicators:

```bash
python -m scripts.monitor_and_retrain --observed data/raw/recent_labeled.csv
```

For local hourly checks with retraining after an accuracy drop, supply an
approved fresh labeled training CSV and opt in explicitly:

```bash
python -m pip install -r requirements-mlflow.txt
python -m scripts.monitor_and_retrain \
  --observed data/raw/recent_labeled.csv \
  --retraining-input data/raw/fresh_training.csv \
  --allow-retraining --interval-minutes 60
```

The scheduling process must remain running; this is **not a hosted
monitoring service**. Retrained models enter the local MLflow registry,
but the serving model does not automatically change without operator
approval. Local data, model files, and reports stay ignored by Git.
See `docs/deployment_monitoring.md` for details and limitations.

The API and monitoring tests use fictional rows. Docker is built in CI
without any private dataset or fitted model. Real-data accuracy, drift
and retraining performance still require separately approved local inputs.
