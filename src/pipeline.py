"""Customer analytics pipeline for the cosmetics e-commerce events dataset.

Reads the monthly event CSVs from data/, cleans them, and writes every
analysis table used by the README and Power BI to outputs/.

Usage:  python src/pipeline.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CHUNK_ROWS = 1_000_000

# Fixed bins: most customers have one order, so quantiles would split ties.
FREQUENCY_BINS = [0, 1, 2, 3, 5, float("inf")]


# --------------------------------------------------------------------------
# Load and clean
# --------------------------------------------------------------------------
def load_and_clean():
    """Stream the raw files; return cleaned purchases, funnel users and QA stats."""
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {DATA_DIR}")

    purchases, qa = [], {"raw_rows": 0, "bad_price": 0, "missing_session": 0}
    funnel_users = {"view": set(), "cart": set(), "purchase": set()}
    event_counts = {}

    for file in files:
        print(f"Reading {file.name}")
        for chunk in pd.read_csv(file, chunksize=CHUNK_ROWS):
            qa["raw_rows"] += len(chunk)
            bad_price = chunk["price"] <= 0
            no_session = chunk["user_session"].isna()
            qa["bad_price"] += int(bad_price.sum())
            qa["missing_session"] += int((~bad_price & no_session).sum())
            chunk = chunk[~bad_price & ~no_session]

            for event, users in chunk.groupby("event_type")["user_id"]:
                funnel_users.setdefault(event, set()).update(users.unique())
                event_counts[event] = event_counts.get(event, 0) + len(users)

            purchases.append(chunk[chunk["event_type"] == "purchase"])

    df = pd.concat(purchases, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates()
    qa["duplicate_purchases"] = before - len(df)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    qa["clean_purchases"] = len(df)
    return df, funnel_users, event_counts, qa


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------
def build_rfm(purchases):
    reference = purchases["event_time"].max() + pd.Timedelta(days=1)
    rfm = purchases.groupby("user_id").agg(
        recency=("event_time", lambda x: (reference - x.max()).days),
        frequency=("user_session", "nunique"),   # orders, not items
        monetary=("price", "sum"),
    )
    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["frequency_score"] = pd.cut(
        rfm["frequency"], bins=FREQUENCY_BINS, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rfm["monetary_score"] = pd.qcut(
        rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rfm["rfm_score"] = (
        rfm["recency_score"].astype(str)
        + rfm["frequency_score"].astype(str)
        + rfm["monetary_score"].astype(str)
    )
    rfm["segment"] = [
        assign_segment(r, f, m)
        for r, f, m in zip(rfm["recency_score"], rfm["frequency_score"], rfm["monetary_score"])
    ]
    return rfm


def assign_segment(r, f, m):
    if r >= 4 and f >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal Customers"
    if r >= 4 and f == 1:
        return "New Customers"
    if r >= 4 and f == 2:
        return "Potential Loyalists"
    if r <= 2 and f >= 3 and m >= 4:
        return "At Risk High Value"
    if r <= 2 and f >= 2:
        return "At Risk"
    if r <= 2 and f == 1 and m >= 3:
        return "High Value Lost"
    return "Regular Customers"


def segment_revenue(rfm):
    out = rfm.groupby("segment").agg(
        customers=("monetary", "count"),
        total_revenue=("monetary", "sum"),
        average_customer_value=("monetary", "mean"),
        average_orders=("frequency", "mean"),
        average_recency_days=("recency", "mean"),
    )
    out["customer_percentage"] = out["customers"] / out["customers"].sum() * 100
    out["revenue_percentage"] = out["total_revenue"] / out["total_revenue"].sum() * 100
    return out.sort_values("total_revenue", ascending=False).reset_index()


def repeat_purchase(purchases):
    out = purchases.groupby("user_id").agg(
        orders=("user_session", "nunique"),
        items_purchased=("user_id", "count"),
        total_spent=("price", "sum"),
    )
    out["customer_type"] = out["orders"].map(lambda n: "One-Time Customer" if n == 1 else "Repeat Customer")
    summary = out.groupby("customer_type").agg(
        customers=("orders", "count"),
        total_revenue=("total_spent", "sum"),
        average_spend=("total_spent", "mean"),
        average_orders=("orders", "mean"),
    )
    summary["customer_percentage"] = summary["customers"] / summary["customers"].sum() * 100
    summary["revenue_percentage"] = summary["total_revenue"] / summary["total_revenue"].sum() * 100
    return out.reset_index(), summary.reset_index()


def monthly_behavior(purchases):
    p = purchases.assign(month=purchases["event_time"].dt.tz_localize(None).dt.to_period("M").astype(str))
    out = p.groupby("month").agg(
        purchasing_customers=("user_id", "nunique"),
        orders=("user_session", "nunique"),
        items_sold=("user_id", "count"),
        total_revenue=("price", "sum"),
    )
    out["revenue_per_order"] = out["total_revenue"] / out["orders"]
    return out.reset_index()


def cohort_retention(purchases):
    p = purchases[["user_id", "event_time"]].copy()
    p["month"] = p["event_time"].dt.tz_localize(None).dt.to_period("M")
    p["cohort"] = p.groupby("user_id")["month"].transform("min")
    p["period"] = (p["month"] - p["cohort"]).apply(lambda d: d.n) + 1
    counts = p.groupby(["cohort", "period"])["user_id"].nunique().unstack()
    retention = counts.div(counts[1], axis=0) * 100
    counts.index = counts.index.astype(str)
    retention.index = retention.index.astype(str)
    return counts, retention.round(2)


def segment_breakdown(purchases, rfm, column, names=None):
    merged = purchases.merge(rfm[["segment"]], left_on="user_id", right_index=True)
    out = (
        merged.groupby(["segment", column], dropna=True)
        .agg(items_sold=(column, "count"), total_sales=("price", "sum"))
        .reset_index()
    )
    if names is not None:
        out["category_name"] = out[column].map(names)
    return out


def purchase_timing(purchases):
    t = purchases["event_time"]
    by_hour = purchases.groupby(t.dt.hour)["price"].agg(items_sold="count", total_sales="sum")
    by_hour.index.name = "hour_utc"
    by_day = purchases.groupby(t.dt.day_name())["price"].agg(items_sold="count", total_sales="sum")
    by_day.index.name = "day_of_week"
    return by_hour.reset_index(), by_day.reset_index()


def funnel_table(funnel_users):
    v, c, p = (len(funnel_users[k]) for k in ("view", "cart", "purchase"))
    return pd.DataFrame({
        "stage": ["Viewed", "Added to cart", "Purchased"],
        "users": [v, c, p],
        "pct_of_viewers": [100, c / v * 100, p / v * 100],
    }).round(2)


# --------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)
    purchases, funnel_users, event_counts, qa = load_and_clean()

    print("\nData quality summary")
    for key, value in qa.items():
        print(f"  {key}: {value:,}")

    rfm = build_rfm(purchases)
    orders, repeat_summary = repeat_purchase(purchases)
    counts, retention = cohort_retention(purchases)
    by_hour, by_day = purchase_timing(purchases)

    names = (
        purchases.dropna(subset=["category_code"])
        .groupby("category_id")["category_code"]
        .agg(lambda x: x.mode().iloc[0])
    )

    tables = {
        "rfm_customer_segments": rfm.reset_index(),
        "segment_revenue_analysis": segment_revenue(rfm),
        "repeat_purchase_analysis": orders,
        "repeat_purchase_summary": repeat_summary,
        "monthly_customer_behavior": monthly_behavior(purchases),
        "cohort_customers": counts.reset_index(),
        "cohort_retention": retention.reset_index(),
        "segment_product_analysis": segment_breakdown(purchases, rfm, "product_id"),
        "segment_brand_analysis": segment_breakdown(purchases, rfm, "brand"),
        "segment_category_analysis": segment_breakdown(purchases, rfm, "category_id", names),
        "purchases_by_hour": by_hour,
        "purchases_by_weekday": by_day,
        "conversion_funnel": funnel_table(funnel_users),
        "data_quality": pd.DataFrame([qa]),
    }
    for name, table in tables.items():
        table.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"Wrote outputs/{name}.csv ({len(table):,} rows)")


if __name__ == "__main__":
    main()
