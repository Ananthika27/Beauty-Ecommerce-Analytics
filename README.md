# Beauty E-Commerce Customer Intelligence

End-to-end customer analytics on **20.7 million shopping events** from a cosmetics online store (October 2019 – February 2020). The project cleans the raw event logs, segments **110,518 buying customers** with RFM analysis, measures repeat purchasing and cohort retention, and turns the findings into retention recommendations. It adds **two predictive models** (next-month repeat purchase for buyers and first purchase for browsers, both validated out of time with confidence intervals), a **brand co-purchase (market-basket) analysis**, an **interactive Streamlit dashboard**, automated tests and CI. Results also feed a Power BI dashboard.

**Tools:** Python · pandas · scikit-learn · Streamlit · matplotlib · SQL · pytest · GitHub Actions · Power BI

## Key findings

1. **Most customers buy once.** 78.9% of customers placed a single order. The 21.1% who repeat generate **48% of revenue** and spend 3.5x more per customer (131 vs 38).
2. **Champions are the most valuable group.** They are 3.5% of customers but **16.9% of revenue**, averaging 277 in spend and 5.8 orders.
3. **The biggest win-back pool is High Value Lost.** 20,189 customers made one high-value order and have been silent for ~118 days. They hold **18.8% of revenue**.
4. **Retention drops sharply after the first month.** Only 18.5% of October's customers bought again in month 2 (the best cohort); later cohorts retain 8.5–9.8%.
5. **Only 6.9% of visitors buy,** and **87% of carts are abandoned** (session level).
6. **Brand preference is very consistent.** `runail` is the top brand in every one of the 8 segments (9.1% of sales overall), followed by `grattol` (7.0%) and `irisk` (5.9%).
7. **Weekdays beat weekends.** Thursday is the strongest sales day and Saturday the weakest; sales peak at 10:00–12:00 UTC.

## Predictive model 1: who among our buyers will buy next month?

**Problem:** rank the 110,518 existing customers by the probability they purchase in the next calendar month, so retention spend goes where it changes behaviour.

**Setup (no leakage):** features use only data *before* a snapshot date; the label is whether the customer bought in the month *after*. The model trains on the Nov, Dec and Jan snapshots and is tested **out of time** on 1 Feb 2020 (92,652 customers, 8.5% base rate). 22 features: 14 purchase-behaviour features (recency, orders, spend, order value, tenure, gaps between orders, brand/category variety, last-30-day activity) plus 8 **browsing features built from the 19.4M view and cart events** that earlier versions ignored (views, carts, cart-to-view ratio, active days, days since last activity).

| Model (test: Feb 2020) | ROC-AUC (95% CI) | PR-AUC | Top-decile lift |
| --- | --- | ---: | ---: |
| Recency/frequency heuristic | 0.714 (0.708–0.719) | 0.258 | 3.0x |
| Logistic regression (all features) | 0.816 (0.811–0.821) | 0.363 | 4.3x |
| Gradient boosting, purchase features only | 0.768 (0.762–0.773) | 0.337 | 4.1x |
| **Gradient boosting + browsing features** | **0.826 (0.822–0.831)** | **0.386** | **4.6x** |

![Model comparison](images/model_comparison.png)
![Deciles](images/model_deciles.png) ![Importance](images/feature_importance.png)

- **Adding browsing data raised AUC from 0.768 to 0.826**, a gain well outside the confidence intervals. The strongest predictor is now *days since the customer last did anything on the site*, not only since they last bought.
- The **top 10% of scored customers return at 39%** versus 8.5% overall, and the **top 20% capture 66% of all returning buyers**.
- Gradient boosting beats logistic regression by only 0.01 AUC (small but outside the intervals); most of the gain comes from the data, not the algorithm.
- **Sanity check:** average predicted probability by segment ranks as expected (Champions 51%, Loyal 27%, Potential Loyalists 19%, High Value Lost 3%).
- **Calibration, and what I could and couldn't fix:** the monthly return rate keeps *falling* (18.5% → 11.3% → 10.8% → 8.5%) as the customer base fills with newer, less loyal buyers, so raw probabilities run too high. Standard isotonic calibration did not help (it assumes a stable rate), so scores are rescaled to a chosen expected rate, which keeps the ranking identical, and the dashboard exposes that rate as a slider. Treat scores as **rankings**, not exact probabilities.

`outputs/customer_retention_scores.csv` scores all customers for March 2020.

## Predictive model 2: which browsers will make a first purchase?

The store has **1.53M users who browsed but never bought**, a group the original analysis ignored. A second model scores them from browsing behaviour alone (1.97M training rows, tested on Feb 2020).

| Model | ROC-AUC (95% CI) | PR-AUC | Top-decile lift |
| --- | --- | ---: | ---: |
| Heuristic (recent carts + views) | 0.791 (0.785–0.798) | 0.043 | 5.4x |
| **Gradient boosting** | **0.852 (0.847–0.857)** | **0.065** | **6.2x** |

![Gains](images/conversion_gains.png)

The **top 10% of browsers contain 62% of next-month first-time buyers** and the top 20% contain 75%, so retargeting spend can be concentrated on a small audience. Note: it only covers people with prior browsing history. Many first-time buyers browse and buy in the same month and cannot be predicted this way.

## Cart abandonment and brand conversion

- Of **983,953 sessions with an item in the cart, 857,691 (87%) ended without a purchase**.
- Cart-to-purchase conversion varies a lot by brand: `severina` converts 34%, `swarovski` 27% and `farmstay` 23%, while `kosmekka` converts only 12% and `babyliss` 12% (`outputs/brand_funnel.csv`, brands with 5,000+ views). Weak converters are candidates for pricing, content or stock review.

## Campaign economics

The dataset has no margin or campaign-cost data, so `src/economics.py` and the dashboard's **Campaign ROI** tab treat them as *adjustable assumptions*. The useful, assumption-light output is the **break-even uplift**: how many extra points of purchase probability a campaign needs to pay for itself. With a 30% margin and a contact cost of 0.30, only High Value Lost customers clear break-even at a 2-point uplift; broad campaigns to Regular and New customers lose money. This is the quantitative case for targeting.

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
| Cart abandoners | Cart-recovery emails; 87% of cart sessions end without a purchase. |
| Browsers who never bought | Retarget only the top-scored 10-20% (they hold 62-75% of next-month first buyers). |
| Campaign timing | Send on Thursdays and mornings (UTC 10–12), when buying peaks. |

## Engineering practices

- **Reproducible:** one command rebuilds every table and chart from the raw files.
- **Tested:** 17 tests cover segment rules, orders-vs-items counting, **leakage** (future purchases or browsing must not change snapshot features), rate rescaling, campaign break-even and basket lift maths; CI runs them on every push.
- **Memory-safe:** the 2.4 GB raw data is streamed in chunks; cleaned purchases are cached as Parquet.
- **Honest evaluation:** out-of-time test set, simple baselines, an ablation (with/without browsing data), bootstrap confidence intervals, and calibration caveats stated.

## Data and method notes

- **Source:** *eCommerce Events History in Cosmetics Shop* (Kaggle), 20,692,840 raw events.
- **Cleaning:** removed 104,288 rows with non-positive prices, 4,597 with no session ID and 898 duplicate purchases, leaving **1,285,982 purchase rows** across 110,518 customers.
- **Orders vs items:** each purchase row is one item, so an *order* is a distinct `user_session`. Counting items would badly overstate repeat buying.
- **Revenue** is the sum of item prices on purchase events (the dataset does not state a currency). Timestamps are UTC.
- **Limitations, and how each was handled:**
  - *Only five months of data:* cannot be fixed without more data (the only source is this Kaggle dataset). Instead, all model results come from a true out-of-time test with bootstrap confidence intervals, and "lost" customers are defined by recency within this window.
  - *Data unused before:* 19.4M view/cart events were ignored. They now drive the browsing features (+0.058 AUC), the browser model, cart-abandonment and brand-conversion analysis.
  - *Categories:* `category_code` is missing for about 96% of revenue and cannot be recovered, so brand is used as the product dimension; price tiers were tested but did not separate segments (about 70% of sales are premium-tier in every segment).
  - *No margin or campaign cost:* replaced by explicit, adjustable assumptions and a break-even calculation, never presented as fact.
  - *No currency stated:* amounts are in the dataset's own price units.
  - *Falling base rate (model scores run high):* rate rescaling plus a slider; scores are for ranking.
  - *Window edge:* customers whose first purchase predates October 2019 appear as "new". Revenue is not profit.

## Project structure

```
├── src/
│   ├── pipeline.py        # clean data + build all analysis tables
│   ├── churn_model.py     # next-month purchase model for buyers
│   ├── conversion_model.py# first-purchase model for browsers
│   ├── economics.py       # campaign break-even / ROI (assumption-driven)
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
make test                  # 17 unit tests
make app                   # interactive dashboard
```

Or run the scripts individually: `python src/pipeline.py`, `python src/churn_model.py`, `python src/conversion_model.py`, `python src/basket.py`, `python src/charts.py`.

## Power BI dashboard

Load the CSVs in `outputs/` into Power BI. Suggested pages: customer overview, RFM segments, revenue by segment, repeat purchase, monthly trend, cohort retention, and brand/product performance by segment.
