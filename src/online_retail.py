"""Run the analytics on the UCI / Kaggle "Online Retail II" dataset (2009-2011, 24 months).

Why: a second, longer dataset to check that the methods generalise, and to validate
lifetime value over a much longer window (including returns).

Expected columns: Invoice, StockCode, Description, Quantity, InvoiceDate, Price,
Customer ID, Country  (older versions: InvoiceNo, UnitPrice, CustomerID).

Usage:  python src/online_retail.py path/to/online_retail_II.csv   (.csv or .xlsx)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import clv
import pipeline

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "online_retail"

RENAMES = {"InvoiceNo": "Invoice", "UnitPrice": "Price", "CustomerID": "Customer ID"}


def load_raw(path):
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)            # Online Retail II has 2 sheets
        raw = pd.concat(sheets.values(), ignore_index=True)
    else:
        raw = pd.read_csv(path, encoding="latin-1")
    return raw.rename(columns=RENAMES)


def to_common_schema(raw):
    """Return (purchases in the pipeline's schema, return statistics)."""
    df = raw.dropna(subset=["Customer ID"]).copy()
    df["Invoice"] = df["Invoice"].astype(str)
    cancelled = df["Invoice"].str.startswith("C")
    line_value = df["Quantity"] * df["Price"]
    returns = {
        "invoices": int(df["Invoice"].nunique()),
        "cancelled_invoices": int(df.loc[cancelled, "Invoice"].nunique()),
        "returned_value": float(-line_value[cancelled].sum()),
        "sales_value": float(line_value[~cancelled & (line_value > 0)].sum()),
    }
    sales = df[~cancelled & (df["Quantity"] > 0) & (df["Price"] > 0)]
    purchases = pd.DataFrame({
        "event_time": pd.to_datetime(sales["InvoiceDate"], utc=True),
        "user_id": sales["Customer ID"].astype(int),
        "user_session": sales["Invoice"],
        "product_id": sales["StockCode"].astype(str),
        "category_id": np.nan,
        "brand": sales["Country"],          # no brand field; country is the closest dimension
        "price": sales["Quantity"] * sales["Price"],
    })
    return purchases.reset_index(drop=True), returns


def main(path):
    OUT.mkdir(parents=True, exist_ok=True)
    purchases, returns = to_common_schema(load_raw(path))
    print(f"{len(purchases):,} sales lines, {purchases['user_id'].nunique():,} customers, "
          f"{purchases['event_time'].min().date()} to {purchases['event_time'].max().date()}")

    rfm = pipeline.build_rfm(purchases)
    orders, repeat_summary = pipeline.repeat_purchase(purchases)
    counts, retention = pipeline.cohort_retention(purchases)
    tables = {
        "rfm_customer_segments": rfm.reset_index(),
        "segment_revenue_analysis": pipeline.segment_revenue(rfm),
        "repeat_purchase_summary": repeat_summary,
        "monthly_customer_behavior": pipeline.monthly_behavior(purchases),
        "cohort_retention": retention.reset_index(),
        "returns_summary": pd.DataFrame([{
            **returns, "return_rate_by_value_pct": round(returns["returned_value"] / returns["sales_value"] * 100, 2)
        }]),
    }

    # CLV validation: fit on the first 18 months, test on the final 6
    tx = clv.transactions(purchases.assign(event_time=purchases["event_time"]))
    end = tx["date"].max()
    metrics, deciles = clv.evaluate_holdout(tx, end - pd.DateOffset(months=6), end)
    tables["clv_validation"] = pd.DataFrame([metrics])
    tables["clv_validation_deciles"] = deciles
    print(metrics)

    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False)
    print(f"Wrote {len(tables)} tables to {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("path", help="Online Retail II csv/xlsx")
    main(parser.parse_args().path)
