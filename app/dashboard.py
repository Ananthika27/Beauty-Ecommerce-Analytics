"""Interactive dashboard.  Run:  streamlit run app/dashboard.py"""
from pathlib import Path

import pandas as pd
import streamlit as st

OUT = Path(__file__).resolve().parent.parent / "outputs"

st.set_page_config(page_title="Beauty E-Commerce Customer Intelligence", layout="wide")


@st.cache_data
def load(name):
    return pd.read_csv(OUT / f"{name}.csv")


st.title("Beauty E-Commerce Customer Intelligence")
st.caption("20.7M shopping events · Oct 2019 – Feb 2020 · 110,518 buying customers")

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

tab_seg, tab_trend, tab_cohort, tab_model, tab_brand = st.tabs(
    ["Segments", "Monthly trend", "Cohorts & funnel", "Retention model", "Brand affinity"]
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

with tab_brand:
    aff = load("brand_affinity")
    min_orders = st.slider("Minimum orders together", 100, 500, 100, 25)
    st.dataframe(aff[aff["orders_together"] >= min_orders].head(50), hide_index=True, width="stretch")
