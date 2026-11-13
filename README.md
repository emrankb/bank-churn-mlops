# Bank churn MLOps

Incremental capstone for preparing banking churn data, training and
tracking predictive models, and deploying a prediction API.

## First milestone: data preparation

The repository currently includes data validation, exploratory summaries,
and an unfitted scikit-learn preprocessing pipeline. Model training,
MLflow tracking, FastAPI, Docker, and monitoring are planned for later
milestones, not implemented yet.

The assignment dataset is `Churn_Modeling.csv` (10,000 records).
**Do not commit it to Git.** Put your local copy at:

```text
data/raw/Churn_Modeling.csv
```

With Python 3.12, install dependencies and run the aggregate analysis:

```bash
python -m pip install -r requirements-dev.txt
python -m scripts.inspect_data
```

The script validates columns, checks the churn label and IDs, and produces
`reports/eda/summary.json` and `reports/eda/churn_by_geography.png`.
The report contains aggregate statistics; both reports and raw data are
ignored by Git.

Run tests without private data:

```bash
python -m pytest -q
```

The preprocessor handles missing numeric and categorical features,
one-hot encodes Geography/Gender, and scales numeric features. It is
created unfitted, so future modeling code can fit it only on training
data. Customer identifiers, surname, and the churn label are not used as
input features.

For implementation choices and dataset notes, see
`docs/data_preparation.md`. This repository uses synthetic test fixtures
in GitHub Actions; actual customer rows never enter the workflow.
