"""Create README charts from the tables in outputs/.  Usage: python src/charts.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT, IMG = ROOT / "outputs", ROOT / "images"
IMG.mkdir(exist_ok=True)
BLUE, GREY = "#2f6fb0", "#b8c0cc"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / name, dpi=150)
    plt.close(fig)


seg = pd.read_csv(OUT / "segment_revenue_analysis.csv").sort_values("revenue_percentage")
fig, ax = plt.subplots(figsize=(8, 4.5))
y = range(len(seg))
ax.barh([i + 0.2 for i in y], seg["revenue_percentage"], 0.4, color=BLUE, label="% of revenue")
ax.barh([i - 0.2 for i in y], seg["customer_percentage"], 0.4, color=GREY, label="% of customers")
ax.set_yticks(list(y)); ax.set_yticklabels(seg["segment"])
ax.set_xlabel("%"); ax.set_title("Customers vs revenue by RFM segment"); ax.legend(frameon=False)
save(fig, "segment_revenue.png")

m = pd.read_csv(OUT / "monthly_customer_behavior.csv")
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(m["month"], m["total_revenue"] / 1e6, color=BLUE)
ax.set_ylabel("Revenue (millions)"); ax.set_title("Monthly revenue")
for i, v in enumerate(m["total_revenue"] / 1e6):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
save(fig, "monthly_revenue.png")

r = pd.read_csv(OUT / "cohort_retention.csv").set_index("cohort_month" if "cohort_month" in pd.read_csv(OUT / "cohort_retention.csv").columns else "cohort")
r = r.drop(columns="1")
fig, ax = plt.subplots(figsize=(7, 3.8))
im = ax.imshow(r.values, cmap="Blues", vmin=0, vmax=20)
ax.set_xticks(range(len(r.columns))); ax.set_xticklabels([f"Month {int(c)}" for c in r.columns])
ax.set_yticks(range(len(r))); ax.set_yticklabels(r.index)
for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        if pd.notna(r.iat[i, j]):
            ax.text(j, i, f"{r.iat[i, j]:.1f}%", ha="center", va="center")
ax.set_title("Cohort retention (% of cohort purchasing again)")
save(fig, "cohort_retention.png")

f = pd.read_csv(OUT / "conversion_funnel.csv")
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.barh(f["stage"][::-1], f["users"][::-1] / 1e6, color=BLUE)
for i, (u, p) in enumerate(zip(f["users"][::-1] / 1e6, f["pct_of_viewers"][::-1])):
    ax.text(u + 0.02, i, f"{u:.2f}M ({p:.1f}%)", va="center")
ax.set_xlabel("Users (millions)"); ax.set_title("Conversion funnel"); ax.set_xlim(0, f["users"].max() / 1e6 * 1.3)
save(fig, "conversion_funnel.png")

h = pd.read_csv(OUT / "purchases_by_hour.csv")
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(h["hour_utc"], h["total_sales"] / 1e3, marker="o", color=BLUE)
ax.set_xlabel("Hour of day (UTC)"); ax.set_ylabel("Sales (thousands)"); ax.set_title("Sales by hour of day")
save(fig, "sales_by_hour.png")
print("Charts saved to images/")

# ---- model charts ----
d = pd.read_csv(OUT / "model_deciles.csv")
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(d["decile"], d["actual_rate"] * 100, color=BLUE, label="Actual return rate")
ax.plot(d["decile"], d["avg_score"] * 100, color="#d9822b", marker="o", label="Predicted")
ax.axhline(d["actual_rate"].mul(d["customers"]).sum() / d["customers"].sum() * 100, color=GREY, ls="--", label="Overall rate")
ax.set_xlabel("Score decile (1 = most likely to buy)"); ax.set_ylabel("% who bought in Feb 2020")
ax.set_title("Out-of-time test: return rate by predicted decile"); ax.legend(frameon=False)
save(fig, "model_deciles.png")

fi = pd.read_csv(OUT / "model_feature_importance.csv").head(8).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(fi["feature"], fi["importance"], xerr=fi["std"], color=BLUE)
ax.set_xlabel("Drop in ROC-AUC when shuffled"); ax.set_title("What drives repeat purchase?")
save(fig, "feature_importance.png")
print("Model charts saved")
