"""Exploratory checks for the credit risk data. Writes reports/eda.json and a few plots."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "credit_risk.csv")
out = ROOT / "reports"
out.mkdir(exist_ok=True)

summary = {
    "rows": int(len(df)),
    "columns": int(df.shape[1]),
    "duplicate_rows": int(df.duplicated().sum()),
    "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
    "class_balance": {str(k): int(v) for k, v in df["loan_status"].value_counts().items()},
    "default_rate": float(df["loan_status"].mean()),
    "person_age_gt_100": int((df["person_age"] > 100).sum()),
    "person_age_max": int(df["person_age"].max()),
    "person_emp_length_gt_60": int((df["person_emp_length"] > 60).sum()),
    "person_emp_length_max": float(df["person_emp_length"].max()),
    "person_income_max": int(df["person_income"].max()),
    "emp_length_gt_age": int((df["person_emp_length"] > df["person_age"] - 14).sum()),
}
(out / "eda.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(df.describe().T.to_string())
for c in df.select_dtypes("object"):
    print(df[c].value_counts().to_string(), "\n")

fig, ax = plt.subplots(2, 3, figsize=(13, 7))
df["loan_status"].value_counts().sort_index().plot.bar(ax=ax[0, 0], title="Class balance")
df["person_age"].clip(upper=100).plot.hist(bins=40, ax=ax[0, 1], title="person_age (clipped at 100)")
df["person_emp_length"].plot.hist(bins=40, ax=ax[0, 2], title="person_emp_length")
df["loan_int_rate"].plot.hist(bins=40, ax=ax[1, 0], title="loan_int_rate")
df.groupby("loan_grade")["loan_status"].mean().plot.bar(ax=ax[1, 1], title="Default rate by grade")
df["person_income"].clip(upper=300000).plot.hist(bins=40, ax=ax[1, 2], title="income (clipped at 300k)")
plt.tight_layout()
plt.savefig(out / "eda.png", dpi=110)
