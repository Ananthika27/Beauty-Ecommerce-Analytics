"""Campaign economics. The dataset has no margin or campaign-cost data, so these
are explicit, adjustable ASSUMPTIONS rather than facts from the data.

  incremental revenue = contacted customers x uplift x average order value
  profit              = incremental revenue x margin - contacted x cost per contact
  break-even uplift   = cost / (margin x average order value)
"""
import pandas as pd


def break_even_uplift(avg_order_value, margin, cost_per_contact):
    """Smallest absolute increase in purchase probability that pays for a contact."""
    return cost_per_contact / (margin * avg_order_value)


def campaign_roi(n_contacted, avg_order_value, uplift, margin, cost_per_contact):
    revenue = n_contacted * uplift * avg_order_value
    cost = n_contacted * cost_per_contact
    profit = revenue * margin - cost
    return {
        "contacted": n_contacted,
        "incremental_orders": n_contacted * uplift,
        "incremental_revenue": revenue,
        "campaign_cost": cost,
        "profit": profit,
        "roi": profit / cost if cost else float("nan"),
    }


def segment_economics(scores, margin=0.30, cost_per_contact=0.30, uplift=0.02):
    """Per-segment campaign economics. `scores` needs segment, avg_order_value."""
    rows = []
    for seg, g in scores.groupby("segment"):
        aov = g["avg_order_value"].mean()
        row = {"segment": seg, "customers": len(g), "avg_order_value": aov,
               "break_even_uplift_pct": break_even_uplift(aov, margin, cost_per_contact) * 100}
        row.update(campaign_roi(len(g), aov, uplift, margin, cost_per_contact))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("profit", ascending=False).round(3)
