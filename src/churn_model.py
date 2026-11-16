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
ENGAGEMENT = ROOT / "data" / "processed" / "engagement_daily.parquet"
OUT = ROOT / "outputs"

TRAIN_SNAPSHOTS = ["2019-11-01", "2019-12-01"]
VALIDATION_SNAPSHOT = "2020-01-01"    # also used for training the final model
TEST_SNAPSHOT = "2020-02-01"
SCORE_SNAPSHOT = "2020-03-01"

FEATURES = [
    "recency_days", "orders", "items", "monetary", "avg_order_value",
    "items_per_order", "tenure_days", "active_months", "distinct_brands",
    "distinct_categories", "avg_days_between_orders", "weekend_share",
    "orders_last_30d", "spend_last_30d",
]
ENGAGEMENT_FEATURES = [
    "views_total", "carts_total", "cart_to_view_ratio", "active_days",
    "views_30d", "carts_30d", "active_days_30d", "days_since_last_activity",
]
ALL_FEATURES = FEATURES + ENGAGEMENT_FEATURES


def load_engagement(path=ENGAGEMENT):
    eng = pd.read_parquet(path)
    eng["date"] = pd.to_datetime(eng["date"])
    return eng


def engagement_features(eng, snapshot):
    """Browsing behaviour (views/carts) before the snapshot, for every user."""
    snap = pd.Timestamp(snapshot)
    past = eng[eng["date"] < snap]
    wide = past.pivot_table(index="user_id", columns="event_type", values="events", aggfunc="sum", fill_value=0)
    feats = pd.DataFrame(index=wide.index)
    feats["views_total"] = wide.get("view", 0)
    feats["carts_total"] = wide.get("cart", 0)
    feats["cart_to_view_ratio"] = feats["carts_total"] / feats["views_total"].clip(lower=1)
    feats["active_days"] = past.groupby("user_id")["date"].nunique()
    feats["days_since_last_activity"] = (snap - past.groupby("user_id")["date"].max()).dt.days
    recent = past[past["date"] >= snap - pd.Timedelta(days=30)]
    rw = recent.pivot_table(index="user_id", columns="event_type", values="events", aggfunc="sum", fill_value=0)
    feats["views_30d"] = rw.get("view", 0)
    feats["carts_30d"] = rw.get("cart", 0)
    feats["active_days_30d"] = recent.groupby("user_id")["date"].nunique()
    for c in ["views_30d", "carts_30d", "active_days_30d"]:
        feats[c] = feats[c].fillna(0)
    return feats[ENGAGEMENT_FEATURES]


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


def make_dataset(purchases, snapshots, eng=None):
    frames = []
    for snap in snapshots:
        X = build_features(purchases, snap)
        if eng is not None:
            X = X.join(engagement_features(eng, snap))
        y = build_labels(purchases, snap, X.index)
        frames.append(X.assign(snapshot=snap, label=y))
    return pd.concat(frames)


def bootstrap_ci(y, p, metric=roc_auc_score, n=200, seed=42):
    """95% bootstrap confidence interval for a metric on the test set."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(metric(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])


def calibration_table(y, p, bins=10):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    return d.groupby("bin").agg(predicted=("p", "mean"), actual=("y", "mean"), customers=("y", "size")).reset_index()


def rescale_to_rate(p, target_rate):
    """Shift probabilities (in logit space) so their mean equals `target_rate`.

    Ranking is unchanged. Use when the expected base rate of the next period
    differs from the training period (here it falls every month as the
    customer base grows with newer, less loyal buyers)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logit + mid)))).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return 1 / (1 + np.exp(-(logit + (lo + hi) / 2)))


def fit_gbm(X, y):
    return HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.06, max_iter=300, l2_regularization=1.0, random_state=42
    ).fit(X, y)




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
    purchases, eng = load_purchases(), load_engagement()

    train = make_dataset(purchases, TRAIN_SNAPSHOTS, eng)
    calib = make_dataset(purchases, [VALIDATION_SNAPSHOT], eng)
    test = make_dataset(purchases, [TEST_SNAPSHOT], eng)
    yte = test["label"]
    print(f"Train {len(train):,} | calibration {len(calib):,} | test {len(test):,} (base rate {yte.mean():.3f})")

    heuristic = -test["recency_days"] + 5 * test["orders"]
    heuristic = (heuristic - heuristic.min()) / (heuristic.max() - heuristic.min())

    both = pd.concat([train, calib])
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(both[ALL_FEATURES].fillna(0), both["label"])
    p_log = baseline.predict_proba(test[ALL_FEATURES].fillna(0))[:, 1]

    purchase_only = fit_gbm(both[FEATURES], both["label"])
    p_purch = purchase_only.predict_proba(test[FEATURES])[:, 1]

    raw_model = fit_gbm(both[ALL_FEATURES], both["label"])
    p_raw = raw_model.predict_proba(test[ALL_FEATURES])[:, 1]
    # Rescale to the PREVIOUS month's observed rate (what we would know in practice)
    p_cal = rescale_to_rate(p_raw, float(calib["label"].mean()))

    rows = []
    for name, p in [
        ("Recency/frequency heuristic", heuristic),
        ("Logistic regression (all features)", p_log),
        ("Gradient boosting (purchase features only)", p_purch),
        ("Gradient boosting (+ browsing features)", p_raw),
        ("Gradient boosting (+ browsing, rate-rescaled)", p_cal),
    ]:
        m = evaluate(name, yte, p)
        lo, hi = bootstrap_ci(yte, p)
        m["roc_auc_ci_low"], m["roc_auc_ci_high"] = round(lo, 4), round(hi, 4)
        rows.append(m)
    metrics = pd.DataFrame(rows)
    print(metrics.to_string(index=False))
    metrics.to_csv(OUT / "model_metrics.csv", index=False)

    decile_table(yte, p_cal).to_csv(OUT / "model_deciles.csv", index=False)
    pd.concat([calibration_table(yte, p_raw).assign(version="before"),
               calibration_table(yte, p_cal).assign(version="after")]).to_csv(OUT / "model_calibration.csv", index=False)

    imp = permutation_importance(raw_model, test[ALL_FEATURES], yte, scoring="roc_auc", n_repeats=5, random_state=42)
    (pd.DataFrame({"feature": ALL_FEATURES, "importance": imp.importances_mean, "std": imp.importances_std})
     .sort_values("importance", ascending=False).round(4).to_csv(OUT / "model_feature_importance.csv", index=False))

    # Final scoring for March: fit on all labelled months, rescale to the latest observed rate
    labelled = pd.concat([train, calib, test])
    final = fit_gbm(labelled[ALL_FEATURES], labelled["label"])
    live = build_features(purchases, SCORE_SNAPSHOT).join(engagement_features(eng, SCORE_SNAPSHOT))
    live["raw_score"] = final.predict_proba(live[ALL_FEATURES])[:, 1]
    live["purchase_probability"] = rescale_to_rate(live["raw_score"], float(yte.mean()))
    live["expected_value_30d"] = live["purchase_probability"] * live["avg_order_value"]
    live["historic_value"] = live["monetary"]
    live.reset_index().to_csv(OUT / "customer_retention_scores.csv", index=False)

    summary = {
        "test_snapshot": TEST_SNAPSHOT,
        "train_rows": int(len(train)), "validation_rows": int(len(calib)), "test_rows": int(len(test)),
        "latest_observed_rate": round(float(yte.mean()), 4),
        "base_rate": round(float(yte.mean()), 4),
    }
    (OUT / "model_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Scored {len(live):,} customers for {SCORE_SNAPSHOT}")


if __name__ == "__main__":
    main()
