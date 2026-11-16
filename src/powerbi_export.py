"""Build a clean star-schema for Power BI from the tables in outputs/.

Writes outputs/powerbi/*.csv. Load that whole folder in Power BI Desktop
(Get Data > Folder, or Text/CSV for each file) and follow dashboard/BUILD_GUIDE.md.

Usage:  python src/powerbi_export.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
PBI = OUT / "powerbi"

SEGMENT_INFO = [
    # segment, sort order, colour, action
    ("Champions", 1, "#1B9E77", "Protect: loyalty rewards, early access, referrals. Avoid heavy discounts."),
    ("Loyal Customers", 2, "#66A61E", "Reward and upsell; push toward Champion behaviour."),
    ("Potential Loyalists", 3, "#7570B3", "Bundles and replenishment reminders to reach a 3rd order."),
    ("New Customers", 4, "#2F6FB0", "Second-order incentive within 30 days."),
    ("Regular Customers", 5, "#999999", "Low-cost nurture only; broad discounts lose money."),
    ("At Risk High Value", 6, "#D95F02", "Personal outreach before they lapse."),
    ("At Risk", 7, "#E6AB02", "Win-back reminders; moderate budget."),
    ("High Value Lost", 8, "#E7298A", "Win-back campaign built around the brand they bought."),
]


def read(name):
    return pd.read_csv(OUT / f"{name}.csv")


def main():
    PBI.mkdir(parents=True, exist_ok=True)

    rfm = read("rfm_customer_segments").rename(columns={"frequency": "orders", "monetary": "total_spent"})
    repeat = read("repeat_purchase_analysis")[["user_id", "items_purchased", "customer_type"]]
    scores = read("customer_retention_scores")[
        ["user_id", "purchase_probability", "avg_order_value", "days_since_last_activity", "views_30d", "carts_30d"]
    ]
    clv = read("customer_clv")[["user_id", "prob_alive", "pred_orders", "pred_value"]].rename(
        columns={"pred_orders": "pred_orders_90d", "pred_value": "pred_value_90d"}
    )
    fact = rfm.merge(repeat, on="user_id", how="left").merge(scores, on="user_id", how="left").merge(clv, on="user_id", how="left")
    fact["probability_band"] = pd.cut(
        fact["purchase_probability"], [-0.01, 0.1, 0.3, 1.0], labels=["Low (<10%)", "Medium (10-30%)", "High (>30%)"]
    ).astype(str)
    fact["score_decile"] = (10 - pd.qcut(fact["purchase_probability"].rank(method="first"), 10, labels=False)).astype(int)
    fact.to_csv(PBI / "fact_customers.csv", index=False)

    pd.DataFrame(SEGMENT_INFO, columns=["segment", "sort_order", "color", "recommended_action"]).to_csv(
        PBI / "dim_segment.csv", index=False
    )

    # Cohort in long format: ideal for a Power BI matrix / heatmap
    counts = read("cohort_customers").melt(id_vars="cohort", var_name="period", value_name="customers")
    pct = read("cohort_retention").melt(id_vars="cohort", var_name="period", value_name="retention_pct")
    cohort = counts.merge(pct, on=["cohort", "period"]).dropna()
    cohort["customers"] = cohort["customers"].astype(int)
    cohort["months_since_first_purchase"] = cohort["period"].astype(int) - 1
    cohort[cohort["customers"] > 0][["cohort", "months_since_first_purchase", "customers", "retention_pct"]].to_csv(
        PBI / "fact_cohort.csv", index=False
    )

    brands = read("segment_brand_analysis")
    brands["rank_in_segment"] = brands.groupby("segment")["total_sales"].rank(ascending=False, method="first").astype(int)
    brands[brands["rank_in_segment"] <= 10].round(2).to_csv(PBI / "fact_segment_brand_top10.csv", index=False)

    weekday = read("purchases_by_weekday")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday["day_number"] = weekday["day_of_week"].map({d: i + 1 for i, d in enumerate(order)})
    weekday.to_csv(PBI / "fact_weekday.csv", index=False)

    direct = {
        "fact_monthly": "monthly_customer_behavior",
        "fact_segment_summary": "segment_revenue_analysis",
        "fact_funnel": "conversion_funnel",
        "fact_hour": "purchases_by_hour",
        "fact_brand_funnel": "brand_funnel",
        "fact_model_metrics": "model_metrics",
        "fact_model_deciles": "model_deciles",
        "fact_repeat_summary": "repeat_purchase_summary",
    }
    for target, source in direct.items():
        table = read(source)
        if target == "fact_monthly":
            table.insert(1, "month_index", range(1, len(table) + 1))   # for month-over-month DAX
        table.to_csv(PBI / f"{target}.csv", index=False)

    read("winback_reasons").to_csv(PBI / "fact_winback_reasons.csv", index=False)
    read("clv_validation_deciles").to_csv(PBI / "fact_clv_validation.csv", index=False)

    read("brand_affinity").head(300).to_csv(PBI / "fact_brand_affinity.csv", index=False)

    for f in sorted(PBI.glob("*.csv")):
        print(f"{f.name:34s} {len(pd.read_csv(f)):>8,} rows")


if __name__ == "__main__":
    main()
