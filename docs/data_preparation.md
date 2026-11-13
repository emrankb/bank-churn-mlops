# Data preparation and initial analysis

The assignment's first task is implemented in `bank_churn/data.py` and
`bank_churn/eda.py`. Place `Churn_Modeling.csv` in the local
`data/raw/` folder (ignored by Git), then run:

```bash
python -m scripts.inspect_data
```

This writes `reports/eda/summary.json` and
`reports/eda/churn_by_geography.png`, also ignored by Git.
The JSON contains counts, rates, and numeric summary statistics, not
customer-level records.

The supplied CSV has 10,000 rows and a pandas-renamed duplicate source
header, `CustomerId.1`. Neither customer identifier nor `Surname`
enters the model features. `Exited` is the binary label, not an input.

The preprocessing factory supplies median imputation and scaling for
numeric columns, plus most-common imputation and one-hot encoding for
categorical columns. The preprocessor is **unfitted** here. Model training
will fit it on training folds only, to avoid leaking the held-out data.

The initial dataset inspection found no missing cells and a churn rate
of 20.37%. This is descriptive dataset analysis, not a model score.
GitHub Actions uses synthetic rows; original customer data is not
committed or uploaded to the CI runners.
