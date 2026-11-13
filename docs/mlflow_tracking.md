# MLflow experiment tracking

Task 3 records the three candidate classifiers from Task 2 and registers the
model selected by mean cross-validated F1.

Run locally with the supplied CSV (ignored by Git):

```bash
python -m pip install -r requirements-dev.txt -r requirements-mlflow.txt
python -m scripts.track_experiments
```

The command evaluates the classifiers through the existing stratified
train/holdout workflow, stores all candidate training-fold CV metrics in
MLflow, and registers the trained winner. It logs the selected model's
held-out accuracy, precision, recall, and F1 in its experiment run. Only
the model chosen from **training-fold CV F1** is registered; the test split
does not determine which model is chosen.

MLflow uses a local SQLite tracking and registry database at
`artifacts/mlflow.db`. The run identifiers, registered version and
`champion` alias are stored in
`artifacts/churn-model/registry.json`. Repeated experiment runs create
new registered versions; the alias points to the most recently selected
candidate. These artifacts are ignored by Git.

MLflow logs parameters such as classifier type, random seed, CV folds and
training/test counts, plus aggregate metrics. No customer identifiers,
individual feature rows, input examples, or individual predictions are
recorded. A serialized fitted pipeline is stored in the local MLflow
artifact store. **Only use local registry databases and model files you
trust**, since model deserialization is code execution.

To view experiments and compare model metrics in the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000` within that environment.
The `bank-churn-classification` experiment shows the three runs, and the
Model Registry displays `bank-churn-classifier` versions and its
`champion` alias.

GitHub Actions uses a synthetic CSV in a temporary directory to validate
the real MLflow integration, without sharing private data or recording
actual customer prediction scores. Real-dataset scores must come from
running the local command on the original supplied dataset.
