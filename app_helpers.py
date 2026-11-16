"""Streamlit-free helpers used by app.py, kept separate so they can be unit tested."""
import json

import joblib
import pandas as pd

import credit_risk as cr

FRIENDLY = {
    "person_age": "Age",
    "person_income": "Annual income",
    "person_emp_length": "Years employed",
    "loan_amnt": "Loan amount",
    "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan as share of income",
    "cb_person_cred_hist_length": "Credit history length",
    "person_home_ownership": "Home ownership",
    "loan_intent": "Loan purpose",
    "loan_grade": "Loan grade",
    "cb_person_default_on_file": "Past default on file",
}


def load_artifacts():
    model = joblib.load(cr.MODEL_DIR / "model.joblib")
    threshold = json.loads((cr.MODEL_DIR / "threshold.json").read_text())["threshold"]
    background = pd.read_csv(cr.MODEL_DIR / "background.csv")
    return model, threshold, background


def validate_inputs(fields: dict) -> list[str]:
    """Return a list of problems with the form values (empty list means fine)."""
    errors = []
    if not 18 <= fields["person_age"] <= cr.MAX_PLAUSIBLE_AGE:
        errors.append(f"Age must be between 18 and {cr.MAX_PLAUSIBLE_AGE}.")
    if fields["person_income"] <= 0:
        errors.append("Income must be above zero.")
    if fields["loan_amnt"] <= 0:
        errors.append("Loan amount must be above zero.")
    if not 0 <= fields["person_emp_length"] <= cr.MAX_PLAUSIBLE_EMP_LENGTH:
        errors.append(f"Years employed must be between 0 and {cr.MAX_PLAUSIBLE_EMP_LENGTH}.")
    if fields["person_emp_length"] > fields["person_age"] - 14:
        errors.append("Years employed is longer than the applicant's working life.")
    for col, allowed in cr.CATEGORY_VALUES.items():
        if fields[col] not in allowed:
            errors.append(f"{col} must be one of {allowed}.")
    return errors


def score(model, threshold: float, frame: pd.DataFrame) -> dict:
    p = float(model.predict_proba(frame)[:, 1][0])
    return {"probability": p, "decision": cr.decide(p, threshold), "threshold": threshold}


def format_reason(reason: dict) -> str:
    name = FRIENDLY.get(reason["feature"], reason["feature"])
    value = reason["value"]
    if isinstance(value, float):
        value = f"{value:g}"
    points = abs(reason["impact"]) * 100
    return f"{name} = {value}: {reason['direction']} ({points:.1f} percentage points)"
