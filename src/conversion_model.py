"""First-purchase model: which browsers who have NEVER bought will buy next month?

The retention model only covers the 110k buyers. The store also has ~1.5M users
who browsed but never purchased; this model scores them from browsing behaviour
(views, carts, activity) using the same leak-free snapshot design.

Train snapshots 2019-11-01 and 2019-12-01, validate 2020-01-01, test 2020-02-01.

Usage:  python src/conversion_model.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import churn_model as cm

OUT = cm.OUT
TRAIN = ["2019-11-01", "2019-12-01", "2020-01-01"]
TEST = "2020-02-01"
SCORE = "2020-03-01"


def dataset(purchases, eng, snapshot):
    snap = pd.Timestamp(snapshot)
    X = cm.engagement_features(eng, snapshot)
    buyers_before = purchases.loc[purchases["event_time"] < snap, "user_id"].unique()
    X = X[~X.index.isin(buyers_before)]                       # never-buyers only
    X = X[X["carts_total"] + X["views_total"] > 0]
    return X


def labelled(purchases, eng, snapshot):
    X = dataset(purchases, eng, snapshot)
    return X.assign(snapshot=snapshot, label=cm.build_labels(purchases, snapshot, X.index))


def main():
    purchases, eng = cm.load_purchases(), cm.load_engagement()
    feats = cm.ENGAGEMENT_FEATURES
    train = pd.concat([labelled(purchases, eng, s) for s in TRAIN])
    test = labelled(purchases, eng, TEST)
    print(f"Train {len(train):,} (rate {train.label.mean():.4f}) | test {len(test):,} (rate {test.label.mean():.4f})")

    model = cm.fit_gbm(train[feats], train["label"])
    p = model.predict_proba(test[feats])[:, 1]
    heuristic = test["carts_30d"] * 3 + test["views_30d"] - test["days_since_last_activity"]
    rows = []
    for name, score in [("Heuristic (recent carts + views)", heuristic.values), ("Gradient boosting", p)]:
        lo, hi = cm.bootstrap_ci(test["label"], score, n=100)
        rows.append({
            "model": name,
            "roc_auc": round(roc_auc_score(test["label"], score), 4),
            "roc_auc_ci_low": round(lo, 4), "roc_auc_ci_high": round(hi, 4),
            "pr_auc": round(average_precision_score(test["label"], score), 4),
            "top_decile_lift": round(cm.top_decile_lift(test["label"].values, np.asarray(score)), 2),
            "base_rate": round(float(test["label"].mean()), 4),
        })
    metrics = pd.DataFrame(rows)
    print(metrics.to_string(index=False))
    metrics.to_csv(OUT / "conversion_model_metrics.csv", index=False)
    cm.decile_table(test["label"], p).to_csv(OUT / "conversion_model_deciles.csv", index=False)

    full = pd.concat([train, test])
    final = cm.fit_gbm(full[feats], full["label"])
    live = dataset(purchases, eng, SCORE)
    live["conversion_probability"] = cm.rescale_to_rate(final.predict_proba(live[feats])[:, 1], float(test["label"].mean()))
    live.reset_index().to_csv(OUT / "browser_conversion_scores.csv", index=False)
    print(f"Scored {len(live):,} never-purchased browsers for {SCORE}")


if __name__ == "__main__":
    main()
