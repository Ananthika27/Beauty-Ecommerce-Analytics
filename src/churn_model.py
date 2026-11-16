"""Predict which customers will buy again next month (retention propensity).

Design (no data leakage):
  * Snapshot dates: features use only purchases BEFORE the snapshot; the label is
    whether the customer bought in the calendar month AFTER the snapshot.
  * Train on snapshots 2019-11-01, 2019-12-01, 2020-01-01.
  * Test out-of-time on snapshot 2020-02-01 (label = purchase in Feb 2020).
    Feb is the last month of data, so the final score for March uses the same
    feature code with snapshot 2020-03-01 and has no label.

Usage:  python src/churn_model.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PURCHASES = ROOT / "data" / "processed" / "purchases.parquet"
OUT = ROOT / "outputs"

TRAIN_SNAPSHOTS = ["2019-11-01", "2019-12-01", "2020-01-01"]
TEST_SNAPSHOT = "2020-02-01"
SCORE_SNAPSHOT = "2020-03-01"

FEATURES = [
    "recency_days", "orders", "items", "monetary", "avg_order_value",
    "items_per_order", "tenure_days", "active_months", "distinct_brands",
    "distinct_categories", "avg_days_between_orders", "weekend_share",
    "orders_last_30d", "spend_last_30d",
]


def load_purchases(path=PURCHASES):
    df = pd.read_parquet(path, columns=["event_time", "user_id", "user_session", "brand", "category_id", "price"])
    df["event_time"] = df["event_time"].dt.tz_localize(None)
    return df


def build_features(purchases, snapshot):
    """One row per customer who bought before `snapshot`, using only past data."""
    snap = pd.Timestamp(snapshot)
    past = purchases[purchases["event_time"] < snap]

    g = past.groupby("user_id")
    orders = g["user_session"].nunique()
    feats = pd.DataFrame({
        "recency_days": (snap - g["event_time"].max()).dt.days,
        "orders": orders,
        "items": g.size(),
        "monetary": g["price"].sum(),
        "tenure_days": (snap - g["event_time"].min()).dt.days,
        "active_months": g["event_time"].apply(lambda s: s.dt.to_period("M").nunique()),
        "distinct_brands": g["brand"].nunique(),
        "distinct_categories": g["category_id"].nunique(),
        "weekend_share": g["event_time"].apply(lambda s: (s.dt.dayofweek >= 5).mean()),
    })
    feats["avg_order_value"] = feats["monetary"] / feats["orders"]
    feats["items_per_order"] = feats["items"] / feats["orders"]

    first_last = past.groupby(["user_id", "user_session"])["event_time"].min().reset_index()
    spans = first_last.groupby("user_id")["event_time"].agg(["min", "max"])
    feats["avg_days_between_orders"] = np.where(
        orders > 1, (spans["max"] - spans["min"]).dt.days / (orders - 1).clip(lower=1), np.nan
    )

    recent = past[past["event_time"] >= snap - pd.Timedelta(days=30)]
    feats["orders_last_30d"] = recent.groupby("user_id")["user_session"].nunique()
    feats["spend_last_30d"] = recent.groupby("user_id")["price"].sum()
    feats[["orders_last_30d", "spend_last_30d"]] = feats[["orders_last_30d", "spend_last_30d"]].fillna(0)
    return feats[FEATURES]


def build_labels(purchases, snapshot, index):
    """1 if the customer bought in the month after the snapshot."""
    start = pd.Timestamp(snapshot)
    end = start + pd.offsets.MonthBegin(1)
    nxt = purchases[(purchases["event_time"] >= start) & (purchases["event_time"] < end)]
    return pd.Series(index.isin(nxt["user_id"].unique()).astype(int), index=index, name="bought_next_month")


def make_dataset(purchases, snapshots):
    frames = []
    for snap in snapshots:
        X = build_features(purchases, snap)
        y = build_labels(purchases, snap, X.index)
        frames.append(X.assign(snapshot=snap, label=y))
    return pd.concat(frames)


def top_decile_lift(y_true, scores):
    n = max(len(y_true) // 10, 1)
    order = np.argsort(-scores)
    return y_true[order][:n].mean() / y_true.mean()


def evaluate(name, y, p):
    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "top_decile_lift": round(top_decile_lift(np.asarray(y), np.asarray(p)), 2),
    }


def decile_table(y, p):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    out = d.groupby("decile", observed=True).agg(customers=("y", "size"), actual_rate=("y", "mean"), avg_score=("p", "mean"))
    out["cumulative_capture"] = (out["actual_rate"] * out["customers"]).cumsum() / d["y"].sum()
    return out.round(4).reset_index()


def main():
    OUT.mkdir(exist_ok=True)
    purchases = load_purchases()

    train = make_dataset(purchases, TRAIN_SNAPSHOTS)
    test = make_dataset(purchases, [TEST_SNAPSHOT])
    Xtr, ytr, Xte, yte = train[FEATURES], train["label"], test[FEATURES], test["label"]
    print(f"Train rows {len(train):,} (base rate {ytr.mean():.3f}) | test rows {len(test):,} (base rate {yte.mean():.3f})")

    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    baseline.fit(Xtr.fillna(0), ytr)
    gbm = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=300, l2_regularization=1.0, random_state=42)
    gbm.fit(Xtr, ytr)

    p_base = baseline.predict_proba(Xte.fillna(0))[:, 1]
    p_gbm = gbm.predict_proba(Xte)[:, 1]
    # Simple non-ML heuristic: more recent + more orders => more likely to return
    heuristic = -Xte["recency_days"] + 5 * Xte["orders"]

    metrics = pd.DataFrame([
        evaluate("Recency/frequency heuristic", yte, (heuristic - heuristic.min()) / (heuristic.max() - heuristic.min())),
        evaluate("Logistic regression", yte, p_base),
        evaluate("Gradient boosting", yte, p_gbm),
    ])
    print(metrics.to_string(index=False))
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    decile_table(yte, p_gbm).to_csv(OUT / "model_deciles.csv", index=False)

    imp = permutation_importance(gbm, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=42)
    (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean, "std": imp.importances_std})
     .sort_values("importance", ascending=False).round(4).to_csv(OUT / "model_feature_importance.csv", index=False))

    # Final scoring: refit on all labelled snapshots, score every customer for March
    full = pd.concat([train, test])
    final = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=300, l2_regularization=1.0, random_state=42)
    final.fit(full[FEATURES], full["label"])
    live = build_features(purchases, SCORE_SNAPSHOT)
    live["purchase_probability"] = final.predict_proba(live[FEATURES])[:, 1]
    live["expected_value_30d"] = live["purchase_probability"] * live["avg_order_value"]
    # High-value customers whose probability is low = best retention targets
    live["historic_value"] = live["monetary"]
    live.reset_index().to_csv(OUT / "customer_retention_scores.csv", index=False)

    summary = {
        "test_snapshot": TEST_SNAPSHOT,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "base_rate": round(float(yte.mean()), 4),
    }
    (OUT / "model_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Scored {len(live):,} customers for {SCORE_SNAPSHOT}")


if __name__ == "__main__":
    main()
