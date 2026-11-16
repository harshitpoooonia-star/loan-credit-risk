import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app_helpers as ah  # noqa: E402
import credit_risk as cr  # noqa: E402


def synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "person_age": rng.integers(20, 60, n),
        "person_income": rng.integers(15000, 150000, n),
        "person_emp_length": rng.integers(0, 20, n).astype(float),
        "loan_amnt": rng.integers(500, 30000, n),
        "loan_int_rate": rng.uniform(5, 22, n),
        "loan_percent_income": rng.uniform(0.02, 0.6, n),
        "cb_person_cred_hist_length": rng.integers(2, 25, n),
        "person_home_ownership": rng.choice(cr.CATEGORY_VALUES["person_home_ownership"], n),
        "loan_intent": rng.choice(cr.CATEGORY_VALUES["loan_intent"], n),
        "loan_grade": rng.choice(cr.CATEGORY_VALUES["loan_grade"], n),
        "cb_person_default_on_file": rng.choice(["N", "Y"], n),
    })
    df["loan_status"] = (df["loan_percent_income"] + rng.normal(0, 0.15, n) > 0.35).astype(int)
    df.loc[:10, "loan_int_rate"] = np.nan
    df.loc[5:15, "person_emp_length"] = np.nan
    return df


# data schema
def test_schema_accepts_good_frame():
    cr.validate_schema(synthetic())


def test_schema_rejects_missing_column():
    with pytest.raises(ValueError):
        cr.validate_schema(synthetic().drop(columns=["loan_grade"]))


def test_schema_rejects_bad_target():
    df = synthetic()
    df.loc[0, "loan_status"] = 2
    with pytest.raises(ValueError):
        cr.validate_schema(df)


def test_clean_drops_duplicates_and_outliers():
    df = synthetic(50)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df.loc[1, "person_age"] = 144
    df.loc[2, "person_emp_length"] = 123.0
    out, info = cr.clean(df)
    assert info["duplicates_dropped"] == 1
    assert info["implausible_rows_dropped"] == 2
    assert len(out) == 51 - 1 - 2
    assert out["person_age"].max() <= 100


def test_real_dataset_shape_if_present():
    if not cr.DATA_PATH.exists():
        pytest.skip("run download_data.py first")
    raw = cr.load_raw()
    assert raw.shape == (32581, 12)
    cr.validate_schema(raw)


# pipeline
@pytest.mark.parametrize("name", ["logreg", "hgb"])
def test_pipeline_fit_predict_shape(name):
    df = synthetic()
    model = cr.build_model(name).fit(df[cr.FEATURES], df[cr.TARGET])
    proba = model.predict_proba(df[cr.FEATURES])
    assert proba.shape == (len(df), 2)
    assert model.predict(df[cr.FEATURES]).shape == (len(df),)


@pytest.mark.parametrize("name", ["logreg", "hgb"])
def test_probability_range_and_sums(name):
    df = synthetic()
    model = cr.build_model(name).fit(df[cr.FEATURES], df[cr.TARGET])
    proba = model.predict_proba(df[cr.FEATURES])
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_pipeline_handles_missing_and_unseen_category():
    df = synthetic()
    model = cr.build_model("logreg").fit(df[cr.FEATURES], df[cr.TARGET])
    row = df[cr.FEATURES].iloc[[0]].copy()
    row["loan_intent"] = "SOMETHING_NEW"
    row["loan_int_rate"] = np.nan
    p = model.predict_proba(row)[:, 1][0]
    assert 0 <= p <= 1


def test_class_weight_is_balanced():
    for name in ["logreg", "hgb"]:
        assert cr.build_model(name).named_steps["clf"].class_weight == "balanced"


# threshold logic
def test_pick_threshold_meets_recall():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 2000), 0, 1)
    t = cr.pick_threshold(y, p, 0.80)
    recall = ((p >= t) & (y == 1)).sum() / (y == 1).sum()
    assert recall >= 0.80


def test_pick_threshold_is_highest_valid():
    y = np.array([1, 1, 1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1])
    assert cr.pick_threshold(y, p, 0.75) == 0.7
    assert cr.pick_threshold(y, p, 1.0) == 0.6


def test_decide_boundary():
    assert cr.decide(0.5, 0.5) == "reject"
    assert cr.decide(0.4999, 0.5) == "approve"
    assert cr.decide(0.0, 0.3) == "approve"
    assert cr.decide(1.0, 0.3) == "reject"


# app helpers
GOOD = dict(
    person_age=30, person_income=60000, person_emp_length=4.0, loan_amnt=9000,
    loan_int_rate=11.0, cb_person_cred_hist_length=5, person_home_ownership="RENT",
    loan_intent="EDUCATION", loan_grade="B", cb_person_default_on_file="N",
)


def test_make_applicant_frame_derives_percent_income():
    f = cr.make_applicant_frame(**GOOD)
    assert list(f.columns) == cr.FEATURES
    assert f.shape == (1, len(cr.FEATURES))
    assert f["loan_percent_income"].iloc[0] == pytest.approx(0.15)


def test_make_applicant_frame_missing_field():
    bad = {k: v for k, v in GOOD.items() if k != "loan_grade"}
    with pytest.raises(ValueError):
        cr.make_applicant_frame(**bad)


def test_validate_inputs():
    assert ah.validate_inputs(GOOD) == []
    assert ah.validate_inputs({**GOOD, "person_age": 150})
    assert ah.validate_inputs({**GOOD, "person_income": 0})
    assert ah.validate_inputs({**GOOD, "person_emp_length": 40.0})
    assert ah.validate_inputs({**GOOD, "loan_grade": "Z"})


def test_score_returns_decision():
    df = synthetic()
    model = cr.build_model("logreg").fit(df[cr.FEATURES], df[cr.TARGET])
    out = ah.score(model, 0.5, cr.make_applicant_frame(**GOOD))
    assert 0 <= out["probability"] <= 1
    assert out["decision"] in {"approve", "reject"}
    assert ah.score(model, 0.0, cr.make_applicant_frame(**GOOD))["decision"] == "reject"
    assert ah.score(model, 1.01, cr.make_applicant_frame(**GOOD))["decision"] == "approve"


def test_encode_decode_roundtrip():
    df = synthetic(20)[cr.FEATURES]
    back = cr.decode_array(cr.encode_frame(df))
    for col in cr.CATEGORICAL:
        assert list(back[col]) == list(df[col])


def test_top_reasons_and_format():
    frame = cr.make_applicant_frame(**GOOD)
    values = np.array([0.01, -0.2, 0.0, 0.05, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    reasons = cr.top_reasons(values, frame, k=3)
    assert [r["feature"] for r in reasons] == ["loan_int_rate", "person_income", "loan_percent_income"]
    assert reasons[1]["direction"] == "lowers risk"
    text = ah.format_reason(reasons[0])
    assert "Interest rate" in text and "raises risk" in text


def test_explain_applicant_shape():
    df = synthetic()
    model = cr.build_model("hgb").fit(df[cr.FEATURES], df[cr.TARGET])
    vals = cr.explain_applicant(model, df[cr.FEATURES].head(50), cr.make_applicant_frame(**GOOD))
    assert vals.shape == (len(cr.FEATURES),)
