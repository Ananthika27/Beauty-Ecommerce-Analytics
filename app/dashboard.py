"""Interactive dashboard.  Run:  streamlit run app/dashboard.py"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import economics as ec
from churn_model import rescale_to_rate

ROOT = Path(__file__).resolve().parent.parent
# Use the full outputs when present, otherwise the small committed demo sample
OUT = ROOT / "outputs" if (ROOT / "outputs" / "rfm_customer_segments.csv").exists() else ROOT / "demo_data"

st.set_page_config(page_title="Beauty E-Commerce Customer Intelligence", layout="wide")


@st.cache_data
def load(name):
    # Full results keep second-store tables in outputs/multicat/; the demo sample prefixes them instead
    if OUT.name != "demo_data":
        for prefix, folder in (("multicat_", "multicat"), ("retail_", "online_retail")):
            if name.startswith(prefix):
                return pd.read_csv(OUT / folder / f"{name[len(prefix):]}.csv")
    return pd.read_csv(OUT / f"{name}.csv")


st.title("Beauty E-Commerce Customer Intelligence")
st.caption("20.7M shopping events · Oct 2019 – Feb 2020 · 110,518 buying customers")
if OUT.name == "demo_data":
    st.info("Demo mode: customer-level tables are a 5,000-customer sample. Aggregate tables are complete.")

seg = load("segment_revenue_analysis")
monthly = load("monthly_customer_behavior")
repeat = load("repeat_purchase_summary")

total_rev = seg["total_revenue"].sum()
repeat_row = repeat.set_index("customer_type").loc["Repeat Customer"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{seg['customers'].sum():,}")
c2.metric("Revenue", f"{total_rev:,.0f}")
c3.metric("Repeat customers", f"{repeat_row['customer_percentage']:.1f}%")
c4.metric("Revenue from repeaters", f"{repeat_row['revenue_percentage']:.1f}%")

tab_seg, tab_trend, tab_cohort, tab_model, tab_clv, tab_uplift, tab_other, tab_roi, tab_browse, tab_brand = st.tabs(
    ["Segments", "Monthly trend", "Cohorts & funnel", "Retention model", "Lifetime value", "Measured uplift", "Other datasets", "Campaign ROI", "Browser conversion", "Brand affinity"]
)

with tab_seg:
    st.subheader("Customers vs revenue by RFM segment")
    chosen = st.multiselect("Segments", seg["segment"].tolist(), default=seg["segment"].tolist())
    view = seg[seg["segment"].isin(chosen)]
    st.bar_chart(view.set_index("segment")[["customer_percentage", "revenue_percentage"]])
    st.dataframe(
        view.round(1).rename(columns={"customer_percentage": "% customers", "revenue_percentage": "% revenue"}),
        hide_index=True, width="stretch",
    )

with tab_trend:
    metric = st.selectbox("Metric", ["total_revenue", "purchasing_customers", "orders", "revenue_per_order"])
    st.line_chart(monthly.set_index("month")[metric])

with tab_cohort:
    left, right = st.columns(2)
    ret = load("cohort_retention").set_index("cohort")
    left.subheader("Cohort retention (%)")
    left.dataframe(ret.drop(columns="1").style.format("{:.1f}", na_rep="").background_gradient(cmap="Blues"))
    funnel = load("conversion_funnel")
    right.subheader("Conversion funnel")
    right.bar_chart(funnel.set_index("stage")["users"])
    right.dataframe(funnel, hide_index=True)

with tab_model:
    metrics = load("model_metrics")
    st.subheader("Predicting who buys next month")
    st.caption("Trained on Nov–Jan snapshots, tested out-of-time on February 2020.")
    st.dataframe(metrics, hide_index=True, width="stretch")
    a, b = st.columns(2)
    a.write("**Return rate by predicted decile**")
    a.bar_chart(load("model_deciles").set_index("decile")["actual_rate"])
    b.write("**Feature importance**")
    b.bar_chart(load("model_feature_importance").head(8).set_index("feature")["importance"])

    st.subheader("Retention target list")
    scores = load("customer_retention_scores").merge(
        load("rfm_customer_segments")[["user_id", "segment"]], on="user_id"
    )
    rate = st.slider(
        "Expected March purchase rate among past buyers", 0.04, 0.20, 0.085, 0.005,
        help="The monthly rate fell from 18% to 8.5% as the customer base grew. Probabilities are rescaled to this rate; the ranking does not change.",
    )
    scores["purchase_probability"] = rescale_to_rate(scores["raw_score"], rate)
    col1, col2 = st.columns(2)
    seg_pick = col1.multiselect("Segment", sorted(scores["segment"].unique()), default=["High Value Lost", "At Risk High Value"])
    max_p = col2.slider("Only customers with purchase probability below", 0.0, 1.0, 0.3, 0.05)
    targets = scores[scores["segment"].isin(seg_pick) & (scores["purchase_probability"] <= max_p)]
    targets = targets.sort_values("historic_value", ascending=False)
    st.write(f"{len(targets):,} customers — historic spend {targets['historic_value'].sum():,.0f}")
    st.dataframe(
        targets[["user_id", "segment", "purchase_probability", "historic_value", "orders", "recency_days"]].head(500),
        hide_index=True, width="stretch",
    )
    st.download_button("Download target list (CSV)", targets.to_csv(index=False), "retention_targets.csv")

    st.subheader("Why are these customers unlikely to return? (SHAP)")
    st.caption("Top 3 factors pulling each customer's purchase probability down.")
    st.dataframe(load("winback_reasons").head(200), hide_index=True, width="stretch")

with tab_clv:
    st.subheader("Customer lifetime value (BG/NBD + Gamma-Gamma)")
    st.caption("Fitted on Oct-Dec 2019 and checked against what customers actually bought in Jan-Feb 2020.")
    val = load("clv_validation").iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Rank correlation vs actual orders", f"{val['correlation_bgnbd']:.2f}", f"naive baseline {val['correlation_naive']:.2f}", delta_color="off")
    k2.metric("Top decile's share of actual spend", f"{val['top_decile_share_of_actual_value_pct']:.0f}%")
    k3.metric("Predicted vs actual orders", f"{int(val['predicted_orders']):,} vs {int(val['actual_orders']):,}")
    st.caption("The model over-predicts total orders because the base keeps filling with newer, less loyal buyers; use it to rank customers by value.")
    v = load("clv_validation_deciles").set_index("decile")[["predicted_value", "actual_value"]]
    st.bar_chart(v)
    clv_tbl = load("customer_clv").merge(load("rfm_customer_segments")[["user_id", "segment"]], on="user_id")
    pick = st.multiselect("Segment ", sorted(clv_tbl["segment"].unique()), default=sorted(clv_tbl["segment"].unique()))
    sel = clv_tbl[clv_tbl["segment"].isin(pick)].sort_values("pred_value", ascending=False)
    st.write(f"Predicted 90-day value of selection: {sel['pred_value'].sum():,.0f}")
    st.dataframe(sel[["user_id", "segment", "prob_alive", "pred_orders", "pred_order_value", "pred_value"]].head(500), hide_index=True, width="stretch")

with tab_uplift:
    st.subheader("What did targeted campaigns actually do? (Dunnhumby Complete Journey)")
    st.caption("A different retailer (grocery, 30 campaigns, 2,500 households) used because the cosmetics data has no campaign records.")
    up_sum = load("dunnhumby_uplift_summary")
    allc = up_sum[up_sum["group"] == "All campaigns"].iloc[0]
    u1, u2, u3 = st.columns(3)
    u1.metric("Naive uplift in spend", f"+{allc['relative_uplift_pct']:.0f}%", f"{allc['avg_uplift_spend_per_household']:.0f} per household", delta_color="off")
    u2.metric("Adjusted for pre-existing trend", f"{allc['adj_relative_uplift_pct']:+.1f}%", f"{allc['adj_uplift_spend_per_household']:+.1f} per household", delta_color="off")
    u3.metric("Buying-probability uplift (naive / adjusted)", f"{allc['avg_uplift_buy_prob_pp']:+.1f} / {allc['adj_uplift_buy_prob_pp']:+.1f} pp")
    st.warning("Targeted households were already spending more before campaigns started (pre-trend check +22.8), so the naive number is inflated. "
               "After adjusting, the evidence for a spend increase is close to zero. Targeting was not randomised: treat this as a range, not a measurement.")
    st.write("**Sanity checks** (both should be near zero for a trustworthy method)")
    st.dataframe(load("dunnhumby_uplift_checks"), hide_index=True, width="stretch")
    st.write("**By campaign type**")
    st.dataframe(up_sum.round(2), hide_index=True, width="stretch")
    st.write("**Each campaign**")
    st.dataframe(load("dunnhumby_campaign_uplift").round(2), hide_index=True, width="stretch")

with tab_other:
    st.subheader("Do the findings hold on a different store?")
    st.caption("Multi-category store (mostly electronics), Oct-Nov 2019, 697,470 customers. Only two months, so some figures are not like-for-like.")
    st.dataframe(load("multicat_generalization_summary").set_index("dataset").T.astype(str), width="stretch")
    st.write("**Do browsing features still help predict next-month purchase?**")
    st.dataframe(load("multicat_generalization_model"), hide_index=True, width="stretch")
    st.caption("Random 50/50 customer split (only two months exist), which is weaker than the out-of-time test used for the cosmetics store.")
    cov = load("multicat_category_coverage").iloc[0, 0]
    st.write(f"**Real category names:** {cov:.1f}% of this store's revenue has a category, versus about 4% for cosmetics.")
    st.dataframe(load("multicat_top_categories"), hide_index=True, width="stretch")

    st.divider()
    st.subheader("A longer window: Online Retail (12 months, wholesale-style customers)")
    st.caption("4,338 customers, Dec 2010 - Dec 2011. Used to validate lifetime value on a 6-month holdout and to measure returns.")
    rep = load("retail_repeat_purchase_summary").set_index("customer_type")
    ret = load("retail_returns_summary").iloc[0]
    cv = load("retail_clv_validation").iloc[0]
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Repeat customers", f"{rep.loc['Repeat Customer', 'customer_percentage']:.0f}%")
    r2.metric("Revenue from repeaters", f"{rep.loc['Repeat Customer', 'revenue_percentage']:.0f}%")
    r3.metric("Returns (by value)", f"{ret['return_rate_by_value_pct']:.1f}%")
    r4.metric("Lifetime-value rank correlation", f"{cv['correlation_bgnbd']:.2f}", f"naive {cv['correlation_naive']:.2f}", delta_color="off")
    st.bar_chart(load("retail_clv_validation_deciles").set_index("decile")[["predicted_value", "actual_value"]])
    st.dataframe(load("retail_segment_revenue_analysis")[["segment", "customers", "customer_percentage", "revenue_percentage"]].round(1), hide_index=True, width="stretch")

with tab_roi:
    st.subheader("Is a retention campaign worth it?")
    st.warning("The cosmetics data has no margin or campaign-cost data. These sliders are assumptions you set. For reference, the Dunnhumby analysis suggests real uplift between about 0 (trend-adjusted) and +2.4 points of buying probability (naive).")
    roi_scores = load("customer_retention_scores").merge(load("rfm_customer_segments")[["user_id", "segment"]], on="user_id")
    a, b, c = st.columns(3)
    margin = a.slider("Gross margin", 0.05, 0.80, 0.30, 0.05)
    cost = b.slider("Cost per customer contacted (same units as price)", 0.05, 3.0, 0.30, 0.05)
    uplift = c.slider("Extra purchase probability from the campaign (points)", 0.5, 10.0, 2.0, 0.5) / 100
    table = ec.segment_economics(roi_scores, margin, cost, uplift)
    st.dataframe(
        table[["segment", "customers", "avg_order_value", "break_even_uplift_pct", "incremental_revenue", "campaign_cost", "profit", "roi"]],
        hide_index=True, width="stretch",
    )
    st.caption("Break-even uplift = the smallest increase in purchase probability that pays for one contact. Segments with a low break-even are the cheapest to win back.")

with tab_browse:
    st.subheader("First-purchase model for browsers who never bought")
    st.dataframe(load("conversion_model_metrics"), hide_index=True, width="stretch")
    conv = load("conversion_model_deciles")
    st.write("**Share of next-month first buyers captured by the top deciles**")
    st.bar_chart(conv.set_index("decile")["cumulative_capture"])
    browsers = load("browser_conversion_scores").sort_values("conversion_probability", ascending=False)
    st.write(f"{len(browsers):,} browsers scored for March 2020")
    st.dataframe(browsers.head(500), hide_index=True, width="stretch")
    st.download_button("Download top browsers (CSV)", browsers.head(20000).to_csv(index=False), "browser_targets.csv")

with tab_brand:
    aff = load("brand_affinity")
    min_orders = st.slider("Minimum orders together", 100, 500, 100, 25)
    st.dataframe(aff[aff["orders_together"] >= min_orders].head(50), hide_index=True, width="stretch")
