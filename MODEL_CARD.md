# Model Card: Loan Credit Risk Predictor

## Summary

HistGradientBoostingClassifier inside a scikit-learn Pipeline that estimates the probability a loan defaults. Built as a learning project.

## Intended use

Demonstrating a credit risk workflow: cleaning, cross-validation, threshold selection and explanations. Not for approving or declining real applicants.

## Data

OpenML 43454 Credit-Risk-Dataset (CC0 per OpenML). After dropping 165 duplicates and 7 implausible rows, 32,409 rows remain. Default rate is about 21.9% in both the training split (25,927 rows) and test split (6,482 rows).

## Inputs

Age, annual income, years employed, loan amount, interest rate, loan as share of income, credit history length, home ownership, loan purpose, loan grade, and past default flag.

## Training

Stratified 80/20 split, 5-fold stratified CV on the training part, `class_weight="balanced"`, seed 42. Model chosen by mean CV ROC-AUC against a logistic regression baseline.

## Decision rule

Reject if predicted default probability is at or above the cutoff. The cutoff is the highest value giving recall of at least 0.80 on out-of-fold training predictions.

## Performance (test set, from reports/metrics.json)

| Metric | Value |
|---|---|
| ROC-AUC | 0.9512 |
| PR-AUC | 0.9112 |
| Precision at cutoff | 0.8210 |
| Recall at cutoff | 0.8216 |
| Cutoff | 0.4677 |

Baseline logistic regression on the same split: ROC-AUC 0.8719, PR-AUC 0.7068.

## Explanations

SHAP permutation values on raw features. Largest average effects: income, loan as share of income, home ownership, interest rate, loan grade, loan purpose.

## Limitations and risks

- One public dataset, one split, no time-based validation.
- No fairness or subgroup analysis. Age and home ownership are model inputs.
- Probabilities were not calibrated after class weighting, so they should not be read as true default rates.
- Applicants outside the training range (for example very high incomes or loan sizes) get unreliable scores.
- Do not use for any real credit decision.
