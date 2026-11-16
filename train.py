"""Train, evaluate and explain the credit risk models. Writes reports/metrics.json."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

import credit_risk as cr

MIN_RECALL = 0.80


def main() -> None:
    cr.MODEL_DIR.mkdir(exist_ok=True)
    cr.REPORT_DIR.mkdir(exist_ok=True)

    df, clean_info = cr.clean(cr.load_raw())
    X, y = df[cr.FEATURES], df[cr.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=cr.SEED
    )
    print(clean_info, "train", len(X_train), "test", len(X_test))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cr.SEED)
    results = {}
    oof = {}
    for name in ["logreg", "hgb"]:
        model = cr.build_model(name)
        p = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        oof[name] = p
        fold_auc, fold_pr = [], []
        for _, va in cv.split(X_train, y_train):
            fold_auc.append(roc_auc_score(y_train.iloc[va], p[va]))
            fold_pr.append(average_precision_score(y_train.iloc[va], p[va]))
        results[name] = {
            "cv_roc_auc_mean": float(np.mean(fold_auc)), "cv_roc_auc_std": float(np.std(fold_auc)),
            "cv_pr_auc_mean": float(np.mean(fold_pr)), "cv_pr_auc_std": float(np.std(fold_pr)),
        }
        print(name, results[name])

    best = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    print("selected model:", best)

    for name in ["logreg", "hgb"]:
        thr = cr.pick_threshold(y_train, oof[name], MIN_RECALL)
        model = cr.build_model(name).fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        results[name].update({
            "threshold": thr,
            "cv_precision_at_threshold": float(precision_score(y_train, oof[name] >= thr)),
            "cv_recall_at_threshold": float(recall_score(y_train, oof[name] >= thr)),
            "test_roc_auc": float(roc_auc_score(y_test, proba)),
            "test_pr_auc": float(average_precision_score(y_test, proba)),
            "test_precision": float(precision_score(y_test, pred)),
            "test_recall": float(recall_score(y_test, pred)),
            "test_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "test_reject_rate": float(pred.mean()),
        })
        if name == best:
            final = model
            final_thr = thr
            final_proba = proba

    metrics = {
        "dataset": "OpenML 43454 Credit-Risk-Dataset",
        "data_cleaning": clean_info,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "train_default_rate": float(y_train.mean()), "test_default_rate": float(y_test.mean()),
        "class_imbalance_handling": "class_weight='balanced' in both models",
        "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on the 80% training split",
        "threshold_rule": f"highest threshold with recall >= {MIN_RECALL} on 5-fold out-of-fold training predictions; reject if probability >= threshold",
        "selected_model": best,
        "selection_rule": "higher mean CV ROC-AUC",
        "models": results,
        "seed": cr.SEED,
        "sklearn_version": sklearn.__version__,
        "shap_version": shap.__version__,
    }

    joblib.dump(final, cr.MODEL_DIR / "model.joblib")
    background = X_train.sample(100, random_state=cr.SEED)
    background.to_csv(cr.MODEL_DIR / "background.csv", index=False)
    (cr.MODEL_DIR / "threshold.json").write_text(json.dumps({"threshold": final_thr, "model": best}, indent=2))

    # SHAP on the 11 raw features, probability scale, so reasons read in applicant terms.
    sample = X_test.sample(500, random_state=cr.SEED)
    f = lambda a: final.predict_proba(cr.decode_array(a))[:, 1]
    masker = shap.maskers.Independent(cr.encode_frame(background), max_samples=100)
    explainer = shap.Explainer(f, masker, algorithm="permutation", seed=cr.SEED)
    sv = explainer(cr.encode_frame(sample), max_evals=2 * len(cr.FEATURES) + 1 + 200, silent=True)
    shap.summary_plot(sv.values, sample, show=False, max_display=11)
    plt.tight_layout()
    plt.savefig(cr.REPORT_DIR / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()
    mean_abs = np.abs(sv.values).mean(axis=0)
    order = np.argsort(-mean_abs)
    metrics["shap_top_features"] = [
        {"feature": cr.FEATURES[i], "mean_abs_shap_probability": float(mean_abs[i])} for i in order
    ]
    metrics["shap_note"] = "permutation explainer, 500 test rows, 100-row background, probability scale"

    (cr.REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
