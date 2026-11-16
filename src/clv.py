"""Customer lifetime value with BG/NBD (purchase counts) + Gamma-Gamma (order value).

Validation: fit on Oct-Dec 2019, then compare predicted vs ACTUAL orders and spend
in the Jan-Feb 2020 holdout. Final model is refit on all data and predicts the next
90 days. Orders are distinct sessions, aggregated to one transaction per customer-day.

Usage:  python src/clv.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

ROOT = Path(__file__).resolve().parent.parent
PURCHASES = ROOT / "data" / "processed" / "purchases.parquet"
OUT = ROOT / "outputs"
PENALIZER = 0.01
HORIZON_DAYS = 90


def transactions(purchases):
    """One row per customer-day with the total order value."""
    t = purchases.copy()
    t["event_time"] = t["event_time"].dt.tz_localize(None)
    t["date"] = t["event_time"].dt.normalize()
    return t.groupby(["user_id", "date"], as_index=False)["price"].sum().rename(columns={"price": "value"})


def fit_models(tx, observation_end):
    summary = summary_data_from_transaction_data(
        tx, "user_id", "date", monetary_value_col="value", observation_period_end=observation_end
    )
    bgf = BetaGeoFitter(penalizer_coef=PENALIZER).fit(summary["frequency"], summary["recency"], summary["T"])
    repeat = summary[summary["frequency"] > 0]
    ggf = GammaGammaFitter(penalizer_coef=PENALIZER).fit(repeat["frequency"], repeat["monetary_value"])
    return bgf, ggf, summary


def predict_clv(bgf, ggf, summary, days):
    out = summary.copy()
    out["pred_orders"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        days, out["frequency"], out["recency"], out["T"]
    )
    out["prob_alive"] = bgf.conditional_probability_alive(out["frequency"], out["recency"], out["T"])
    # Gamma-Gamma needs a repeat purchase; fall back to the customer's single order value
    expected_value = pd.Series(np.nan, index=out.index)
    mask = out["frequency"] > 0
    expected_value[mask] = ggf.conditional_expected_average_profit(out.loc[mask, "frequency"], out.loc[mask, "monetary_value"])
    expected_value[~mask] = out.loc[~mask, "monetary_value"].replace(0, np.nan).fillna(out["monetary_value"].mean())
    out["pred_order_value"] = expected_value
    out["pred_value"] = out["pred_orders"] * out["pred_order_value"]
    return out


def evaluate_holdout(tx, calib_end, hold_end):
    """Fit on data up to calib_end; compare predicted vs actual in (calib_end, hold_end]."""
    holdout_days = (hold_end - calib_end).days
    bgf, ggf, summary = fit_models(tx[tx["date"] <= calib_end], calib_end)
    pred = predict_clv(bgf, ggf, summary, holdout_days)
    actual = tx[(tx["date"] > calib_end) & (tx["date"] <= hold_end)].groupby("user_id").agg(
        actual_orders=("date", "count"), actual_value=("value", "sum")
    )
    pred = pred.join(actual).fillna({"actual_orders": 0, "actual_value": 0})
    naive = pred["frequency"] / pred["T"].clip(lower=1) * holdout_days
    pred["decile"] = pd.qcut(pred["pred_value"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    deciles = pred.groupby("decile", observed=True).agg(
        customers=("pred_value", "size"), predicted_value=("pred_value", "sum"), actual_value=("actual_value", "sum")
    ).reset_index()
    deciles["actual_share_pct"] = deciles["actual_value"] / deciles["actual_value"].sum() * 100
    metrics = {
        "holdout_days": holdout_days,
        "predicted_orders": round(pred["pred_orders"].sum()),
        "actual_orders": int(pred["actual_orders"].sum()),
        "correlation_bgnbd": round(np.corrcoef(pred["pred_orders"], pred["actual_orders"])[0, 1], 4),
        "correlation_naive": round(np.corrcoef(naive, pred["actual_orders"])[0, 1], 4),
        "top_decile_share_of_actual_value_pct": round(float(deciles.loc[0, "actual_share_pct"]), 2),
    }
    return metrics, deciles.round(2)


def main():
    OUT.mkdir(exist_ok=True)
    purchases = pd.read_parquet(PURCHASES, columns=["event_time", "user_id", "price"])
    tx = transactions(purchases)

    metrics, deciles = evaluate_holdout(tx, pd.Timestamp("2019-12-31"), pd.Timestamp("2020-02-29"))
    print(metrics)
    print(deciles.round(1).to_string(index=False))
    deciles.to_csv(OUT / "clv_validation_deciles.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUT / "clv_validation.csv", index=False)

    # Final: refit on everything and predict the next 90 days
    end = tx["date"].max()
    bgf, ggf, summary = fit_models(tx, end)
    final = predict_clv(bgf, ggf, summary, HORIZON_DAYS).reset_index()
    final[["user_id", "frequency", "recency", "T", "prob_alive", "pred_orders", "pred_order_value", "pred_value"]].round(4).to_csv(
        OUT / "customer_clv.csv", index=False
    )
    print(f"Predicted {HORIZON_DAYS}-day value for {len(final):,} customers; total {final['pred_value'].sum():,.0f}")


if __name__ == "__main__":
    main()
