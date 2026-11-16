"""Market-basket analysis: which brands are bought together in the same order?

An order = one user_session. For each brand pair (A, B) we report
  support    = share of multi-item orders containing both
  confidence = P(B in order | A in order)
  lift       = support / (P(A) * P(B))   (>1 means bought together more than chance)

Usage:  python src/basket.py
"""
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PURCHASES = ROOT / "data" / "processed" / "purchases.parquet"
OUT = ROOT / "outputs"
MIN_PAIR_ORDERS = 100


def brand_pairs(purchases, min_pair_orders=MIN_PAIR_ORDERS):
    baskets = (
        purchases.dropna(subset=["brand"])
        .groupby("user_session")["brand"]
        .agg(lambda s: tuple(sorted(set(s))))
    )
    baskets = baskets[baskets.map(len) >= 2]
    n_orders = len(baskets)

    single, pair = Counter(), Counter()
    for items in baskets:
        single.update(items)
        pair.update(combinations(items, 2))

    rows = []
    for (a, b), c in pair.items():
        if c < min_pair_orders:
            continue
        support = c / n_orders
        rows.append({
            "brand_a": a, "brand_b": b, "orders_together": c,
            "support": support,
            "confidence_a_to_b": c / single[a],
            "confidence_b_to_a": c / single[b],
            "lift": support / ((single[a] / n_orders) * (single[b] / n_orders)),
        })
    out = pd.DataFrame(rows).sort_values("lift", ascending=False)
    return out.round(4).reset_index(drop=True), n_orders


def main():
    OUT.mkdir(exist_ok=True)
    purchases = pd.read_parquet(PURCHASES, columns=["user_session", "brand"])
    pairs, n = brand_pairs(purchases)
    pairs.to_csv(OUT / "brand_affinity.csv", index=False)
    print(f"{n:,} multi-brand orders; {len(pairs)} brand pairs with >= {MIN_PAIR_ORDERS} joint orders")
    print(pairs.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
