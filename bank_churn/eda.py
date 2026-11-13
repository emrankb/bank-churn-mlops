"""Exploratory summaries. Outputs contain only aggregate data."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data import CATEGORICAL, NUMERIC, TARGET, validate_churn_data


def summarize_churn(frame: pd.DataFrame) -> dict:
    data = validate_churn_data(frame)
    grouped = {}
    for field in CATEGORICAL:
        values = (
            data.groupby(field, dropna=False)[TARGET]
            .agg(["size", "sum", "mean"])
            .reset_index()
        )
        grouped[field] = [
            {
                "category": str(row[field]),
                "customers": int(row["size"]),
                "churned": int(row["sum"]),
                "churn_rate": round(float(row["mean"]), 6),
            }
            for _, row in values.iterrows()
        ]
    numeric_stats = data.loc[:, NUMERIC].describe().round(4).to_dict()
    return {
        "rows": int(len(data)),
        "churned": int(data[TARGET].sum()),
        "churn_rate": round(float(data[TARGET].mean()), 6),
        "missing_by_column": {key: int(v) for key, v in data.isna().sum().items()},
        "grouped": grouped,
        "numeric_summary": numeric_stats,
    }


def write_eda(frame: pd.DataFrame, output_dir: str | Path) -> dict:
    """Write local, aggregate-only stats and a churn-rate chart."""
    summary = summarize_churn(frame)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows = summary["grouped"]["Geography"]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar([r["category"] for r in rows], [100 * r["churn_rate"] for r in rows])
    axis.set_ylabel("Churn rate (%)")
    axis.set_title("Customer churn by geography")
    fig.tight_layout()
    fig.savefig(dest / "churn_by_geography.png", dpi=150)
    plt.close(fig)
    return summary
