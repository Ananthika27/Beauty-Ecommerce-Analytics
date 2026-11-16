# Beauty E-Commerce Customer Intelligence

End-to-end customer analytics on **20.7 million shopping events** from a cosmetics online store (October 2019 – February 2020). The project cleans the raw event logs, segments **110,518 buying customers** with RFM analysis, measures repeat purchasing and cohort retention, and turns the findings into retention recommendations. It adds a **predictive model for next-month purchase**, a **brand co-purchase (market-basket) analysis**, an **interactive Streamlit dashboard**, automated tests and CI. Results also feed a Power BI dashboard.

**Tools:** Python · pandas · scikit-learn · Streamlit · matplotlib · SQL · pytest · GitHub Actions · Power BI

## Key findings

1. **Most customers buy once.** 78.9% of customers placed a single order. The 21.1% who repeat generate **48% of revenue** and spend 3.5x more per customer (131 vs 38).
2. **Champions are the most valuable group.** They are 3.5% of customers but **16.9% of revenue**, averaging 277 in spend and 5.8 orders.
3. **The biggest win-back pool is High Value Lost.** 20,189 customers made one high-value order and have been silent for ~118 days. They hold **18.8% of revenue**.
4. **Retention drops sharply after the first month.** Only 18.5% of October's customers bought again in month 2 (the best cohort); later cohorts retain 8.5–9.8%.
5. **Only 6.9% of visitors buy.** 24.9% of viewers add to cart, and about 28% of cart users go on to purchase.
6. **Brand preference is very consistent.** `runail` is the top brand in every one of the 8 segments (9.1% of sales overall), followed by `grattol` (7.0%) and `irisk` (5.9%).
7. **Weekdays beat weekends.** Thursday is the strongest sales day and Saturday the weakest; sales peak at 10:00–12:00 UTC.

## Predictive model: who will buy next month?

**Problem:** rank customers by the probability they purchase in the next calendar month, so retention spend goes where it changes behaviour.

**Setup (no leakage):** features are built only from purchases *before* a snapshot date; the label is whether the customer bought in the month *after*. The model trains on the Nov, Dec and Jan snapshots (150,510 customer-snapshots) and is tested **out of time** on the 1 Feb 2020 snapshot (92,652 customers, 8.5% base rate). 14 behavioural features cover recency, order count, spend, average order value, tenure, gaps between orders, brand/category variety, weekend share and last-30-day activity.

| Model (test: Feb 2020) | ROC-AUC | PR-AUC | Brier | Top-decile lift |
| --- | ---: | ---: | ---: | ---: |
| Recency/frequency heuristic | 0.714 | 0.258 | 0.075 | 3.0x |
| Logistic regression | 0.765 | 0.323 | 0.071 | 4.0x |
| **Gradient boosting** | **0.768** | **0.337** | **0.068** | **4.1x** |

![Deciles](images/model_deciles.png) ![Importance](images/feature_importance.png)

- The **top 10% of scored customers return at 35%** versus 8.5% overall, and the top 20% capture **57% of all returning buyers**. A campaign sent to the top two deciles reaches over half the people who will buy anyway; the bottom five deciles hold just 20%.
- **Recency is by far the strongest driver**, then the typical gap between a customer's orders and spend in the last 30 days.
- The gradient-boosting model beats a simple heuristic by 0.05 AUC but is only marginally better than logistic regression. The signal here is mostly behavioural recency, and I report that rather than overstate the gain.
- **Sanity check:** average predicted probability by segment ranks as expected: Champions 53%, Loyal 28%, Potential Loyalists 21%, High Value Lost 3%.
- **Caveat:** scores run slightly above the observed rate on the test month (February was quieter than the training months), so use them for *ranking*, not as calibrated absolute probabilities.

`outputs/customer_retention_scores.csv` scores all 110,518 customers for March 2020. The dashboard turns this into a downloadable **retention target list**. For example, 20,884 "High Value Lost" and "At Risk High Value" customers have under a 10% purchase probability and about 1.31M in historic spend.

## Market-basket analysis

Across 91,831 multi-brand orders, brand pairs bought together far more than chance (lift > 1) include `eunyul` + `farmstay` (lift 40, 374 orders), `airnails` + `jas` (lift 30, 327 orders) and `estelare` + `shary` (lift 29). These pairs suggest bundles and cross-sell recommendations; lift, support and confidence for 879 pairs are in `outputs/brand_affinity.csv`.

## Interactive dashboard

```bash
streamlit run app/dashboard.py
```

Tabs: segments (filterable), monthly trend, cohorts and funnel, retention model with a filterable and downloadable target list, and brand affinity.

## Charts

| | |
| --- | --- |
| ![Segments](images/segment_revenue.png) | ![Monthly revenue](images/monthly_revenue.png) |
| ![Cohorts](images/cohort_retention.png) | ![Funnel](images/conversion_funnel.png) |

![Sales by hour](images/sales_by_hour.png)

## RFM customer segments

Customers are scored 1–5 on **Recency** (days since last order), **Frequency** (number of *orders*) and **Monetary** (total spend). Recency and monetary use quintiles; frequency uses fixed bins (1, 2, 3, 4–5, 6+ orders) because most customers have exactly one order and quantile splits would divide identical customers.

| Segment | Customers | % Customers | % Revenue | Avg spend | Avg orders | Days since last order |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regular Customers | 39,586 | 35.8% | 18.0% | 29 | 1.1 | 93 |
| New Customers | 30,508 | 27.6% | 18.8% | 39 | 1.0 | 26 |
| High Value Lost | 20,189 | 18.3% | 18.8% | 59 | 1.0 | 118 |
| Potential Loyalists | 7,060 | 6.4% | 9.4% | 85 | 2.0 | 25 |
| Loyal Customers | 4,456 | 4.0% | 10.3% | 146 | 3.2 | 38 |
| At Risk | 3,930 | 3.6% | 5.3% | 86 | 2.0 | 110 |
| Champions | 3,877 | 3.5% | 16.9% | 277 | 5.8 | 19 |
| At Risk High Value | 912 | 0.8% | 2.5% | 171 | 3.5 | 105 |

| Segment | Rule (R, F, M scores) |
| --- | --- |
| Champions | R ≥ 4, F ≥ 4 |
| Loyal Customers | R ≥ 3, F ≥ 3 |
| New Customers | R ≥ 4, F = 1 |
| Potential Loyalists | R ≥ 4, F = 2 |
| At Risk High Value | R ≤ 2, F ≥ 3, M ≥ 4 |
| At Risk | R ≤ 2, F ≥ 2 |
| High Value Lost | R ≤ 2, F = 1, M ≥ 3 |
| Regular Customers | everyone else |

## Monthly behavior

| Month | Customers | Orders | Revenue | Revenue / order |
| --- | ---: | ---: | ---: | ---: |
| 2019-10 | 25,762 | 29,326 | 1,210,922 | 41.29 |
| 2019-11 | 31,524 | 36,568 | 1,530,831 | 41.86 |
| 2019-12 | 25,613 | 28,894 | 1,077,689 | 37.30 |
| 2020-01 | 28,220 | 32,385 | 1,321,825 | 40.82 |
| 2020-02 | 25,759 | 29,401 | 1,207,001 | 41.05 |

November (Black Friday season) is the peak month for customers and revenue; December is the weakest, with the lowest revenue per order.

## Recommendations

| Segment | Action |
| --- | --- |
| Champions | Protect them: early access, loyalty rewards, referral offers. Avoid heavy discounting. |
| High Value Lost | Win-back campaign (email/SMS) built around the brand they bought, such as `runail` and `grattol`. |
| At Risk High Value | Personal outreach before they lapse; they already have 3+ orders. |
| New Customers | Second-order incentive within 30 days. Retention falls off sharply after month 1. |
| Potential Loyalists | Bundles and replenishment reminders to push them to a 3rd order. |
| Cart abandoners | Cart-recovery emails; roughly 72% of cart users never purchase. |
| Campaign timing | Send on Thursdays and mornings (UTC 10–12), when buying peaks. |

## Engineering practices

- **Reproducible:** one command rebuilds every table and chart from the raw files.
- **Tested:** 14 tests cover segment rules, orders-vs-items counting, **leakage** (adding future purchases must not change snapshot features) and basket lift maths; CI runs them on every push.
- **Memory-safe:** the 2.4 GB raw data is streamed in chunks; cleaned purchases are cached as Parquet.
- **Honest evaluation:** out-of-time test set, comparison to simple baselines, calibration caveat stated.

## Data and method notes

- **Source:** *eCommerce Events History in Cosmetics Shop* (Kaggle), 20,692,840 raw events.
- **Cleaning:** removed 104,288 rows with non-positive prices, 4,597 with no session ID and 898 duplicate purchases, leaving **1,285,982 purchase rows** across 110,518 customers.
- **Orders vs items:** each purchase row is one item, so an *order* is a distinct `user_session`. Counting items would badly overstate repeat buying.
- **Revenue** is the sum of item prices on purchase events (the dataset does not state a currency). Timestamps are UTC.
- **Limitations:** five months of data, so cohort and "lost" signals are short-window; `category_code` is missing for most purchases, so categories are shown by numeric `category_id`; there is no discount or margin data, so revenue is not profit.

## Project structure

```
├── src/
│   ├── pipeline.py        # clean data + build all analysis tables
│   ├── churn_model.py     # next-month purchase model (out-of-time validated)
│   ├── basket.py          # brand co-purchase / market-basket analysis
│   └── charts.py          # charts for this README
├── app/dashboard.py       # Streamlit dashboard
├── tests/                 # pytest suite (segments, leakage, labels, basket)
├── .github/workflows/     # CI runs the tests on every push
├── Makefile
├── notebooks/
│   └── 01_data_exploration.ipynb
├── sql/customer_analysis.sql
├── images/                # charts used above
├── outputs/               # generated tables (git-ignored; used by Power BI)
├── data/                  # raw monthly CSVs (git-ignored)
├── dashboard/             # Power BI files
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
# download the dataset from Kaggle and place the five monthly CSVs in data/
make all                   # pipeline -> model -> basket -> charts (a few minutes)
make test                  # 14 unit tests
make app                   # interactive dashboard
```

Or run the scripts individually: `python src/pipeline.py`, `python src/churn_model.py`, `python src/basket.py`, `python src/charts.py`.

## Power BI dashboard

Load the CSVs in `outputs/` into Power BI. Suggested pages: customer overview, RFM segments, revenue by segment, repeat purchase, monthly trend, cohort retention, and brand/product performance by segment.
