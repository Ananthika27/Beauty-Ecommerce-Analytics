"""Explain the retention model with SHAP.

* Global: which features move predictions most (images/shap_summary.png).
* Local: for win-back targets, the top reasons each customer is unlikely to buy
  (outputs/winback_reasons.csv), so a marketer can see WHY, not just a score.

Usage:  python src/explain.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import churn_model as cm

ROOT = cm.ROOT
OUT, IMG = cm.OUT, ROOT / "images"
N_GLOBAL, N_LOCAL = 1500, 3000


def main():
    IMG.mkdir(exist_ok=True)
    purchases, eng = cm.load_purchases(), cm.load_engagement()
    snaps = cm.TRAIN_SNAPSHOTS + [cm.VALIDATION_SNAPSHOT]
    train = cm.make_dataset(purchases, snaps, eng)
    test = cm.make_dataset(purchases, [cm.TEST_SNAPSHOT], eng)
    feats = cm.ALL_FEATURES
    model = cm.fit_gbm(train[feats], train["label"])

    def predict(X):
        return model.predict_proba(pd.DataFrame(X, columns=feats))[:, 1]

    background = train[feats].sample(100, random_state=0)
    explainer = shap.Explainer(predict, background, algorithm="permutation", seed=0)

    sample = test[feats].sample(N_GLOBAL, random_state=0)
    values = explainer(sample, max_evals=2 * len(feats) + 1)
    pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(values.values).mean(axis=0)}).sort_values(
        "mean_abs_shap", ascending=False
    ).round(5).to_csv(OUT / "shap_importance.csv", index=False)
    plt.figure()
    shap.summary_plot(values.values, sample, show=False, max_display=12)
    plt.title("How each feature moves a customer's purchase probability (SHAP)")
    plt.tight_layout()
    plt.savefig(IMG / "shap_summary.png", dpi=150)
    plt.close()

    # Local reasons for the win-back list
    scores = pd.read_csv(OUT / "customer_retention_scores.csv").merge(
        pd.read_csv(OUT / "rfm_customer_segments.csv")[["user_id", "segment"]], on="user_id"
    )
    targets = scores[scores["segment"].isin(["High Value Lost", "At Risk High Value"])]
    targets = targets.sort_values("historic_value", ascending=False).head(N_LOCAL)
    X = targets[feats]
    local = explainer(X, max_evals=2 * len(feats) + 1)
    rows = []
    for i, (_, r) in enumerate(targets.iterrows()):
        order = np.argsort(local.values[i])[:3]           # most negative = biggest drag
        reasons = [f"{feats[j]}={X.iloc[i, j]:.4g}" for j in order]
        rows.append({
            "user_id": int(r["user_id"]), "segment": r["segment"],
            "purchase_probability": round(r["purchase_probability"], 4),
            "historic_value": round(r["historic_value"], 2),
            "main_reason_1": reasons[0], "main_reason_2": reasons[1], "main_reason_3": reasons[2],
        })
    pd.DataFrame(rows).to_csv(OUT / "winback_reasons.csv", index=False)
    print("Saved shap_importance.csv, winback_reasons.csv and images/shap_summary.png")


if __name__ == "__main__":
    main()
