"""Streamlit app: enter an applicant, get default probability, decision and top reasons."""
import streamlit as st

import app_helpers as ah
import credit_risk as cr

st.set_page_config(page_title="Loan Credit Risk Predictor", layout="centered")
st.title("Loan Credit Risk Predictor")
st.caption("Student project on a public dataset. Not for real lending decisions.")


@st.cache_resource
def get_artifacts():
    return ah.load_artifacts()


try:
    model, threshold, background = get_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run `python train.py` first.")
    st.stop()

with st.form("applicant"):
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 18, 100, 30)
    income = c2.number_input("Annual income", 1000, 10_000_000, 55000, step=1000)
    emp = c1.number_input("Years employed", 0.0, 60.0, 4.0, step=0.5)
    amnt = c2.number_input("Loan amount", 500, 100_000, 8000, step=500)
    rate = c1.number_input("Interest rate (%)", 1.0, 40.0, 11.0, step=0.1)
    hist = c2.number_input("Credit history length (years)", 0, 60, 4)
    home = c1.selectbox("Home ownership", cr.CATEGORY_VALUES["person_home_ownership"])
    intent = c2.selectbox("Loan purpose", cr.CATEGORY_VALUES["loan_intent"])
    grade = c1.selectbox("Loan grade", cr.CATEGORY_VALUES["loan_grade"], index=1)
    default_on_file = c2.selectbox("Past default on file", cr.CATEGORY_VALUES["cb_person_default_on_file"])
    go = st.form_submit_button("Score applicant")

if go:
    fields = dict(
        person_age=age, person_income=income, person_emp_length=emp, loan_amnt=amnt,
        loan_int_rate=rate, cb_person_cred_hist_length=hist, person_home_ownership=home,
        loan_intent=intent, loan_grade=grade, cb_person_default_on_file=default_on_file,
    )
    problems = ah.validate_inputs(fields)
    if problems:
        for p in problems:
            st.error(p)
    else:
        frame = cr.make_applicant_frame(**fields)
        result = ah.score(model, threshold, frame)
        st.metric("Default probability", f"{result['probability']:.1%}")
        st.write(f"Loan as share of income: {frame['loan_percent_income'].iloc[0]:.0%}")
        if result["decision"] == "reject":
            st.error(f"Reject (probability is at or above the {threshold:.3f} cutoff)")
        else:
            st.success(f"Approve (probability is below the {threshold:.3f} cutoff)")
        with st.spinner("Working out the main reasons"):
            values = cr.explain_applicant(model, background, frame)
        st.subheader("Top reasons")
        for r in cr.top_reasons(values, frame, k=3):
            st.write("- " + ah.format_reason(r))
        st.caption("Reasons are SHAP contributions to the default probability, measured against a sample of training applicants.")
