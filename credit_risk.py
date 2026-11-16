"""Shared code: data loading, cleaning, pipeline, threshold logic, explanation helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "credit_risk.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SEED = 42

TARGET = "loan_status"
NUMERIC = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
CATEGORICAL = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]
FEATURES = NUMERIC + CATEGORICAL
EXPECTED_COLUMNS = FEATURES + [TARGET]

CATEGORY_VALUES = {
    "person_home_ownership": ["RENT", "MORTGAGE", "OWN", "OTHER"],
    "loan_intent": ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"],
    "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
    "cb_person_default_on_file": ["N", "Y"],
}

MAX_PLAUSIBLE_AGE = 100
MAX_PLAUSIBLE_EMP_LENGTH = 60


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if not set(df[TARGET].dropna().unique()) <= {0, 1}:
        raise ValueError("loan_status must be 0 or 1")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicates and impossible ages / employment lengths."""
    validate_schema(df)
    n0 = len(df)
    df = df.drop_duplicates()
    n_dup = n0 - len(df)
    bad = (df["person_age"] > MAX_PLAUSIBLE_AGE) | (df["person_emp_length"] > MAX_PLAUSIBLE_EMP_LENGTH)
    n_bad = int(bad.sum())
    df = df.loc[~bad].reset_index(drop=True)
    info = {
        "rows_raw": int(n0),
        "duplicates_dropped": int(n_dup),
        "implausible_rows_dropped": n_bad,
        "rows_used": int(len(df)),
    }
    return df[EXPECTED_COLUMNS], info


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def build_model(name: str) -> Pipeline:
    if name == "logreg":
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    elif name == "hgb":
        clf = HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=300, learning_rate=0.08,
            early_stopping=False, random_state=SEED,
        )
    else:
        raise ValueError(f"unknown model: {name}")
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def pick_threshold(y_true, proba, min_recall: float = 0.80) -> float:
    """Highest threshold whose recall on the given predictions is >= min_recall.

    Higher threshold means fewer flagged applicants, so this gives the best precision
    that still catches at least min_recall of the defaulters.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    best = None
    for t in np.unique(proba)[::-1]:
        flagged = proba >= t
        recall = (flagged & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)
        if recall >= min_recall:
            best = float(t)
            break
    if best is None:
        best = float(proba.min())
    return best


def decide(probability: float, threshold: float) -> str:
    """Reject when the default probability is at or above the threshold."""
    return "reject" if probability >= threshold else "approve"


def make_applicant_frame(**fields) -> pd.DataFrame:
    """One-row frame in training column order. loan_percent_income is derived if absent."""
    row = dict(fields)
    if "loan_percent_income" not in row:
        income = row.get("person_income") or 0
        row["loan_percent_income"] = round(row["loan_amnt"] / income, 4) if income else np.nan
    missing = [c for c in FEATURES if c not in row]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    return pd.DataFrame([row])[FEATURES]


def top_reasons(shap_values: np.ndarray, frame: pd.DataFrame, k: int = 3) -> list[dict]:
    """Largest absolute SHAP contributions for one applicant (probability scale)."""
    order = np.argsort(-np.abs(shap_values))[:k]
    out = []
    for i in order:
        col = frame.columns[i]
        out.append({
            "feature": col,
            "value": frame.iloc[0, i],
            "impact": float(shap_values[i]),
            "direction": "raises risk" if shap_values[i] > 0 else "lowers risk",
        })
    return out


def encode_frame(df: pd.DataFrame) -> np.ndarray:
    """Numeric array for SHAP maskers: categoricals become index codes (-1 if unseen)."""
    out = df[FEATURES].copy()
    for col, values in CATEGORY_VALUES.items():
        out[col] = out[col].map({v: i for i, v in enumerate(values)}).fillna(-1)
    return out.to_numpy(dtype=float)


def decode_array(arr: np.ndarray) -> pd.DataFrame:
    """Inverse of encode_frame, used inside the SHAP prediction function."""
    df = pd.DataFrame(np.atleast_2d(arr), columns=FEATURES)
    for col, values in CATEGORY_VALUES.items():
        idx = df[col].round().astype(int).clip(0, len(values) - 1)
        df[col] = idx.map(dict(enumerate(values)))
    return df


def explain_applicant(model, background: pd.DataFrame, frame: pd.DataFrame, seed: int = SEED) -> np.ndarray:
    """SHAP values (probability scale) for a one-row applicant frame."""
    import shap

    f = lambda a: model.predict_proba(decode_array(a))[:, 1]
    masker = shap.maskers.Independent(encode_frame(background), max_samples=100)
    explainer = shap.Explainer(f, masker, algorithm="permutation", seed=seed)
    return explainer(encode_frame(frame), max_evals=2 * len(FEATURES) + 1 + 200, silent=True).values[0]
