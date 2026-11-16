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

# ---- model comparison with confidence intervals ----
mm = pd.read_csv(OUT / "model_metrics.csv").iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.barh(mm["model"], mm["roc_auc"], color=[BLUE if "browsing" in n else GREY for n in mm["model"]],
        xerr=[mm["roc_auc"] - mm["roc_auc_ci_low"], mm["roc_auc_ci_high"] - mm["roc_auc"]], capsize=3)
ax.set_xlim(0.6, 0.9); ax.set_xlabel("ROC-AUC on Feb 2020 (95% bootstrap CI)")
ax.set_title("Browsing data lifts next-month purchase prediction")
save(fig, "model_comparison.png")

cd = pd.read_csv(OUT / "conversion_model_deciles.csv")
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(cd["decile"], cd["cumulative_capture"] * 100, marker="o", color=BLUE, label="Model")
ax.plot([0] + list(cd["decile"]), [0] + [i * 10 for i in cd["decile"]], color=GREY, ls="--", label="Random")
ax.set_xlabel("Browsers contacted, by score decile"); ax.set_ylabel("% of next-month first buyers reached")
ax.set_title("Cumulative gains: never-purchased browsers"); ax.legend(frameon=False)
save(fig, "conversion_gains.png")
print("Comparison charts saved")

# ---- CLV validation ----
cv = pd.read_csv(OUT / "clv_validation_deciles.csv")
fig, ax = plt.subplots(figsize=(7.5, 4))
w = 0.4
ax.bar(cv["decile"] - w / 2, cv["predicted_value"] / 1e3, w, color=GREY, label="Predicted")
ax.bar(cv["decile"] + w / 2, cv["actual_value"] / 1e3, w, color=BLUE, label="Actual")
ax.set_xlabel("Predicted-value decile (1 = highest)"); ax.set_ylabel("Spend in Jan-Feb 2020 (thousands)")
ax.set_title("Lifetime value model: predicted vs actual spend (holdout)"); ax.legend(frameon=False)
save(fig, "clv_validation.png")
print("CLV chart saved")

# ---- measured campaign uplift (Dunnhumby) ----
up = pd.read_csv(OUT / "dunnhumby_campaign_uplift.csv")
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(up["pretrend_spend"], up["uplift_spend_per_household"], s=up["targeted_households"] / 6 + 15, color=BLUE, alpha=0.7)
lims = [min(up["pretrend_spend"].min(), up["uplift_spend_per_household"].min()), max(up["pretrend_spend"].max(), up["uplift_spend_per_household"].max())]
ax.plot(lims, lims, color="#d9822b", ls="--", label="Uplift = pre-existing trend (true effect 0)")
ax.set_xlabel("Targeted households' extra spend BEFORE the campaign")
ax.set_ylabel("Targeted households' extra spend DURING the campaign")
ax.set_xlim(-130, 140); ax.set_ylim(-130, 140)
ax.set_title("Campaign uplift vs pre-existing trend (29 campaigns; 3 outliers off-scale)"); ax.legend(frameon=False, loc="upper left")
save(fig, "campaign_uplift.png")

lib = pd.read_csv(OUT / "model_library_comparison.csv")
fig, ax = plt.subplots(figsize=(7, 3.2))
x = range(len(lib))
ax.bar([i - 0.2 for i in x], lib["test_roc_auc_default"], 0.4, color=GREY, label="Default settings")
ax.bar([i + 0.2 for i in x], lib["test_roc_auc_tuned"], 0.4, color=BLUE, label="Tuned")
ax.set_xticks(list(x)); ax.set_xticklabels(lib["model"]); ax.set_ylim(0.8, 0.84); ax.set_ylabel("ROC-AUC (Feb 2020)")
ax.set_title("Algorithm choice barely matters"); ax.legend(frameon=False)
save(fig, "library_comparison.png")
print("Uplift + library charts saved")

# ---- Online Retail: lifetime value validation on a 6-month holdout ----
retail = OUT / "online_retail" / "clv_validation_deciles.csv"
if retail.exists():
    rv = pd.read_csv(retail)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(rv["decile"] - 0.2, rv["predicted_value"] / 1e3, 0.4, color=GREY, label="Predicted")
    ax.bar(rv["decile"] + 0.2, rv["actual_value"] / 1e3, 0.4, color=BLUE, label="Actual")
    ax.set_xlabel("Predicted-value decile (1 = highest)"); ax.set_ylabel("Spend in final 6 months (thousands)")
    ax.set_title("Online Retail: lifetime value predicted vs actual (holdout)"); ax.legend(frameon=False)
    save(fig, "retail_clv_validation.png")
    print("Retail chart saved")
