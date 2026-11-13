# Model training and evaluation

Task 2 compares multiple churn classifiers and chooses a model by F1-score.
Use `python -m scripts.train_models` after placing the supplied CSV locally
at `data/raw/Churn_Modeling.csv`.

## Evaluation approach

- Validate the raw schema, customer IDs, numeric fields, and binary `Exited` target.
- Exclude `CustomerId`, `CustomerId.1`, and `Surname` from the features.
- Hold out 20% of records using a stratified, reproducible split.
- On the remaining 80%, evaluate a majority-class baseline, logistic
  regression, and random forest with 3-fold stratified cross-validation.
- Include imputation, scaling, and one-hot encoding **inside each pipeline**,
  ensuring transformers are fitted only on the training portion of a fold.
- Select the highest mean cross-validated F1 model (stable tie-breaking).
- Refit that model on the full training split and calculate accuracy,
  precision, recall, and F1 once on the untouched holdout.

This is more reliable than selecting on the held-out test F1 and reporting
that same score as an unbiased final evaluation. The primary selection
criterion is still F1, as required by the assignment.

Outputs under `artifacts/churn-model/` are ignored by Git:

- `model.joblib`: fitted preprocessing-plus-classifier pipeline
- `metrics.json`: aggregate cross-validation and held-out metrics, split
  sizes, feature schema, and random seed

Neither generated model artifacts nor private customer data are committed.
Only load a joblib artifact generated from a trusted local source; the
format is pickle-based and can execute code when loaded.

The CI suite trains on invented rows. Passing tests proves the training
process works; **it is not a real-dataset model performance score**.
A local run with the original dataset is required before reporting model
accuracy or F1. Future milestones will log experiments in MLflow and expose
an approved selected model through an API.
