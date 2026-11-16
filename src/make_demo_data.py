"""Create a small, committable sample of outputs/ so the dashboard can be deployed
(e.g. Streamlit Community Cloud) without the 2.4 GB raw data.

Aggregate tables are copied in full; customer-level tables are sampled.
Usage:  python src/make_demo_data.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT, DEMO = ROOT / "outputs", ROOT / "demo_data"
N_CUSTOMERS = 5000

AGGREGATES = [
    "segment_revenue_analysis", "monthly_customer_behavior", "repeat_purchase_summary", "cohort_retention",
    "conversion_funnel", "model_metrics", "model_deciles", "model_feature_importance", "brand_affinity",
    "conversion_model_metrics", "conversion_model_deciles", "clv_validation", "clv_validation_deciles",
    "shap_importance", "brand_funnel",
]
CUSTOMER_LEVEL = ["rfm_customer_segments", "customer_retention_scores", "customer_clv", "winback_reasons"]


def main():
    DEMO.mkdir(exist_ok=True)
    for name in AGGREGATES:
        pd.read_csv(OUT / f"{name}.csv").to_csv(DEMO / f"{name}.csv", index=False)

    ids = pd.read_csv(OUT / "rfm_customer_segments.csv")["user_id"].sample(N_CUSTOMERS, random_state=0)
    for name in CUSTOMER_LEVEL:
        table = pd.read_csv(OUT / f"{name}.csv")
        keep = table[table["user_id"].isin(ids)] if name != "winback_reasons" else table.head(300)
        keep.to_csv(DEMO / f"{name}.csv", index=False)

    browsers = pd.read_csv(OUT / "browser_conversion_scores.csv").nlargest(2000, "conversion_probability")
    browsers.to_csv(DEMO / "browser_conversion_scores.csv", index=False)

    size = sum(f.stat().st_size for f in DEMO.glob("*.csv")) / 1e6
    print(f"Wrote {len(list(DEMO.glob('*.csv')))} files to demo_data/ ({size:.1f} MB)")


if __name__ == "__main__":
    main()
