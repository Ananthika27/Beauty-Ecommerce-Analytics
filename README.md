# Beauty E-Commerce Customer Intelligence 📊

An end-to-end customer analytics project built using Python and Power BI to understand customer purchasing behavior, customer value, retention, and revenue patterns in an e-commerce cosmetics store.

## Project Overview

This project analyzes the **eCommerce Events History in Cosmetics Shop** dataset, which contains customer, product, brand, category, price, and purchase-event information from **October 2019 to February 2020**.

The project focuses on understanding customer behavior and answering questions such as:

- Who are the most valuable customers?
- Which customers are likely to stop purchasing?
- How much revenue comes from different customer segments?
- How often do customers make repeat purchases?
- Which products, brands, and categories are preferred by different customer segments?
- How does customer retention change over time?

## Tools Used

**Python | Pandas | NumPy | Jupyter Notebook | SQL | Power BI | GitHub**

## Analysis

### RFM Customer Segmentation

Customers were segmented using **Recency, Frequency and Monetary (RFM)** analysis.

The analysis identified 8 customer segments:

| Customer Segment | Customers | % of Customers | % of Revenue |
| --- | --- | --- | --- |
| Regular Customers | 39,586 | 35.82% | 17.99% |
| New Customers | 30,508 | 27.60% | 18.82% |
| High Value Lost | 20,189 | 18.27% | 18.83% |
| Potential Loyalists | 7,060 | 6.39% | 9.42% |
| Loyal Customers | 4,456 | 4.03% | 10.27% |
| At Risk | 3,930 | 3.56% | 5.32% |
| Champions | 3,877 | 3.51% | 16.90% |
| At Risk High Value | 912 | 0.83% | 2.45% |

Frequency is the number of **orders** (distinct sessions), scored with fixed bins (1, 2, 3, 4–5, 6+ orders) because most customers have a single order. Recency and monetary use quintiles.

### Revenue Analysis

The customer segments showed a clear difference in their contribution to revenue.

**High Value Lost generated 18.83% of total revenue**, followed by:

- New Customers — **18.82%**
- Regular Customers — **17.99%**
- Champions — **16.90%**

Champions are only 3.5% of customers but average 276 in spend per customer, versus 39 for New Customers.

### Repeat Purchase Analysis

Customer purchase frequency was analyzed to understand repeat purchasing behavior.

A purchase is counted as one order (distinct `user_session`), not one item.

- **23,303 customers (21.1%)** placed more than one order.
- **87,215 customers (78.9%)** placed a single order.

Repeat customers generated 48.04% of revenue.

The analysis also looked at the distribution of customer purchase frequency.

### Monthly Customer Behavior

Customer behavior and revenue were analyzed across the five months in the dataset.

| Month | Purchasing Customers | Revenue |
| --- | --- | --- |
| Oct 2019 | 25,762 | 1,210,921.55 |
| Nov 2019 | 31,524 | 1,530,831.06 |
| Dec 2019 | 25,613 | 1,077,689.23 |
| Jan 2020 | 28,220 | 1,321,825.06 |
| Feb 2020 | 25,759 | 1,207,000.80 |

**November 2019 recorded the highest revenue** during the analyzed period.

### Cohort Retention Analysis

Customer cohorts were created based on the month of their first purchase to understand how many customers continued purchasing in subsequent months.

The October 2019 cohort showed the highest observed retention across the available cohort periods, with:

- Month 2 — **18.49%**
- Month 3 — **12.78%**
- Month 4 — **13.19%**
- Month 5 — **10.45%**

### Product, Brand & Category Analysis

Customer segments were further analyzed based on their purchasing preferences.

The analysis identified the top-performing:

- Products
- Brands
- Categories

for each customer segment.

> **Data limitation:** `category_code` is missing for ~96% of purchase revenue in the source data, so categories are reported by numeric `category_id`; a readable `category_name` is attached only where available.

This helped connect **customer value with product preferences** and provides a basis for more targeted customer strategies.

## Key Insights

- **Champions contribute 16.90% of revenue** from 3.5% of customers.
- **78.9% of customers placed only one order**; repeat customers (21.1%) generate 48.04% of revenue.
- **High Value Lost customers (one order, high spend, inactive 90+ days) contribute 18.83% of revenue** — the biggest win-back opportunity.
- **November 2019 generated the highest monthly revenue** at approximately 1.53M.
- Customer preferences vary across different RFM segments, with different products, brands, and categories performing better within different customer groups.
- Cohort analysis shows that customer retention drops significantly after the first month, highlighting the importance of retaining newly acquired customers.

## Power BI Dashboard

The analysis was converted into an interactive Power BI dashboard to make the results easier to explore.

The dashboard covers:

- Customer overview
- RFM customer segmentation
- Revenue by customer segment
- Customer purchase frequency
- Monthly customer behavior
- Cohort retention
- Product performance
- Brand performance
- Category performance

## Project Structure

```
Beauty-Ecommerce-Analytics/
│
├── data/
├── dashboard/
├── notebooks/
├── sql/
│
├── load_data.py
├── requirements.txt
├── .gitignore
└── README.md
```

The raw dataset and generated CSV files are excluded from the GitHub repository using `.gitignore` because of their size.

## How to Run

```bash
pip install -r requirements.txt
# place the five monthly CSVs from the Kaggle dataset in data/
python load_data.py
```

The script cleans the data (drops duplicates, non-positive prices, missing sessions), then writes the analysis CSVs (including `cohort_retention.csv`) used by Power BI.

## Outcome

This project helped me work through an end-to-end customer analytics workflow — from exploring and processing raw e-commerce event data in Python to performing customer segmentation and retention analysis, and finally presenting the results through Power BI.

It combines **customer analytics, business analysis, data visualization, and Python-based data processing** in one project.