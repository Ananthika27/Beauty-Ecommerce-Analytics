"""Do the cosmetics findings hold on a second store?

Runs on the multi-category store (Oct-Nov 2019) after
`python src/pipeline.py --data-dir data_external/multicat --out-dir outputs/multicat`.

1. Descriptive comparison (repeat rate, revenue concentration, funnel, cart abandonment).
2. Model check: does adding browsing features lift next-month purchase prediction here too?
   Only two months exist, so a time split is impossible. Customers are split 50/50 at
   random, which is WEAKER than the out-of-time test used for the cosmetics store.
3. Real category names: category_code coverage and top categories.

Usage:  python src/generalization_check.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import churn_model as cm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
MULTI = OUT / "multicat"
PROC = ROOT / "data_external" / "multicat" / "processed"


def share(df, col, value):
    return float(df.loc[df[col] == value, "customer_percentage"].iloc[0])


def descriptive():
    rows = []
    for name, folder in [("Cosmetics store (5 months)", OUT), ("Multi-category store (2 months)", MULTI)]:
        rep = pd.read_csv(folder / "repeat_purchase_summary.csv")
        seg = pd.read_csv(folder / "segment_revenue_analysis.csv")
        fun = pd.read_csv(folder / "conversion_funnel.csv").set_index("stage")
        qa = pd.read_csv(folder / "data_quality.csv").iloc[0]
        top = seg.sort_values("revenue_percentage", ascending=False).iloc[0]
        rows.append({
            "dataset": name,
            "customers": int(seg["customers"].sum()),
            "single_order_pct": round(share(rep, "customer_type", "One-Time Customer"), 1),
            "repeat_revenue_share_pct": round(float(rep.loc[rep["customer_type"] == "Repeat Customer", "revenue_percentage"].iloc[0]), 1),
            "top_segment_by_revenue": top["segment"],
            "top_segment_revenue_pct": round(top["revenue_percentage"], 1),
            "view_to_cart_users_pct": round(float(fun.loc["Added to cart", "pct_of_viewers"]), 1),
            "view_to_purchase_users_pct": round(float(fun.loc["Purchased", "pct_of_viewers"]), 1),
            "abandoned_cart_sessions_pct": round(qa["abandoned_cart_sessions"] / qa["cart_sessions"] * 100, 1),
        })
    return pd.DataFrame(rows)


def model_check():
    purchases = cm.load_purchases(PROC / "purchases.parquet")
    eng = cm.load_engagement(PROC / "engagement_daily.parquet")
    data = cm.make_dataset(purchases, ["2019-11-01"], eng)
    rng = np.random.default_rng(0)
    is_train = rng.random(len(data)) < 0.5
    train, test = data[is_train], data[~is_train]
    rows = []
    for name, feats in [("Purchase features only", cm.FEATURES), ("+ browsing features", cm.ALL_FEATURES)]:
        gbm = cm.fit_gbm(train[feats], train["label"])
        p = gbm.predict_proba(test[feats])[:, 1]
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(train[feats].fillna(0), train["label"])
        lo, hi = cm.bootstrap_ci(test["label"], p, n=100)
        rows.append({
            "features": name, "gbm_roc_auc": round(roc_auc_score(test["label"], p), 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "logistic_roc_auc": round(roc_auc_score(test["label"], lr.predict_proba(test[feats].fillna(0))[:, 1]), 4),
            "top_decile_lift": round(cm.top_decile_lift(test["label"].values, p), 2),
            "base_rate": round(float(test["label"].mean()), 4), "test_customers": len(test),
        })
    return pd.DataFrame(rows)


def categories():
    p = pd.read_parquet(PROC / "purchases.parquet", columns=["category_code", "price"])
    covered = p["category_code"].notna()
    top = p[covered].assign(group=lambda d: d["category_code"].str.split(".").str[:2].str.join("."))
    table = top.groupby("group")["price"].agg(items_sold="count", revenue="sum").sort_values("revenue", ascending=False)
    table["share_of_covered_revenue_pct"] = table["revenue"] / table["revenue"].sum() * 100
    cover = pd.DataFrame([{"revenue_with_category_pct": round(p.loc[covered, "price"].sum() / p["price"].sum() * 100, 1)}])
    return table.head(15).round(2).reset_index(), cover


def main():
    desc = descriptive()
    desc.to_csv(MULTI / "generalization_summary.csv", index=False)
    print(desc.T.to_string(header=False))
    mc = model_check()
    mc.to_csv(MULTI / "generalization_model.csv", index=False)
    print(mc.to_string(index=False))
    cats, cover = categories()
    cats.to_csv(MULTI / "top_categories.csv", index=False)
    cover.to_csv(MULTI / "category_coverage.csv", index=False)
    print(cover.to_string(index=False)); print(cats.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
