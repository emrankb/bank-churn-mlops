# FastAPI, Docker, monitoring and retraining

This implements the final MLOps assignment task with a **local-only**
deployment and a scheduled monitoring process. It is not a production
bank service: no authentication, encryption termination, access-control
layer or live banking data integration is included.

## Train a model and serve predictions

With the private `Churn_Modeling.csv` saved locally in `data/raw/`:

```bash
python -m pip install -r requirements-dev.txt -r requirements-api.txt
python -m scripts.train_models
python -m scripts.serve_api
```

The service listens at `127.0.0.1:8000` by default. Use
`GET /health` for readiness and `POST /predict` for one customer's
feature values. A prediction request has ten fields: CreditScore,
Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard,
IsActiveMember and EstimatedSalary. Identifiers, surnames and the
`Exited` outcome are forbidden. The response contains only a 0/1
prediction and class-1 estimated probability.

The API loads the saved fitted pipeline **once at start** and refuses
to start if the trusted local model file is missing. Loading a joblib
file is code execution: **never** load an untrusted uploaded model.
No prediction requests or records are persisted by the API. Avoid
publishing the service directly on the internet.

## Docker

Build a container without embedding customer data or the model:

```bash
docker build -t bank-churn-mlops .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v "$(pwd)/artifacts/churn-model:/app/artifacts/churn-model:ro" \
  bank-churn-mlops
```

The image runs as a non-root user. The trained model is mounted
read-only from the host at runtime and stays outside the image and
the Git repository. `GET http://127.0.0.1:8000/health` checks the
container once it starts. `http://127.0.0.1:8000/docs` provides
the FastAPI request schema. The image has no built-in training data.

## Monitoring

Monitoring needs a **separate recent, labeled batch** with the same
CSV schema and real known `Exited` outcomes. Predicting churn alone
does not give a later true churn label. Run a one-time check:

```bash
python -m scripts.monitor_and_retrain \
  --observed data/raw/recent_labeled.csv
```

The monitor compares observed accuracy to the original held-out
test accuracy in `artifacts/churn-model/metrics.json`. A drop of at
least 0.05 (absolute accuracy points) recommends retraining.
It also calculates basic numeric mean shifts relative to the reference
standard deviation and categorical total-variation shifts. These
**simulated drift indicators are not statistical tests** and drift alone
does not trigger retraining.

All reports are aggregated and saved under `reports/monitoring/`,
which is ignored by Git. No individual records, predictions or
identifiers appear in monitoring outputs.

## Local scheduled retraining

A process that stays running can periodically evaluate the newest
labeled batch. To allow retraining on accuracy degradation, a separate
**approved, fresh labeled training dataset** is required:

```bash
python -m pip install -r requirements-dev.txt -r requirements-mlflow.txt
python -m scripts.monitor_and_retrain \
  --observed data/raw/recent_labeled.csv \
  --retraining-input data/raw/fresh_training.csv \
  --allow-retraining --interval-minutes 60 --cooldown-hours 24
```

The command checks once per hour **only while this process is running**.
If measured accuracy falls by at least the specified threshold, it
trains candidate models on the separate approved dataset, selects by
training-fold CV F1, logs an MLflow experiment and creates a new
registered version. A 24-hour cooldown prevents repeated retraining
for every subsequent alert. The model serving current requests is not
automatically replaced: deployment/promotion requires explicit
operator review and restarting the service with an approved model.

For a durable schedule outside a running terminal, a trusted
operator can run the command via cron or a process supervisor.
GitHub Actions does **not** schedule private-data retraining: no
private CSV is stored in the repository or uploaded to CI. The
simulated CI monitoring and retraining tests use invented records.

## Limitations

The model's estimated probability is not a guaranteed individual
outcome or a causal explanation. Historical `Exited` is a training
label. Monitoring without later confirmed churn labels cannot measure
accuracy. Retraining is triggered from aggregate accuracy checks,
not by arbitrary data drift. Drift thresholds are heuristic and
should be tuned against real operational data before production.
