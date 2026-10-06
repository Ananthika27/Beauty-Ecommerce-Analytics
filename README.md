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

| Customer Segment | Customers | % of Customers |
| --- | --- | --- |
| Regular Customers | 35,658 | 32.26% |
| Champions | 16,470 | 14.90% |
| At Risk | 14,479 | 13.10% |
| Loyal Customers | 12,562 | 11.37% |
| New Customers | 11,142 | 10.08% |
| At Risk High Value | 10,338 | 9.35% |
| High Value Lost | 5,440 | 4.92% |
| Potential Loyalists | 4,429 | 4.01% |

### Revenue Analysis

The customer segments showed a clear difference in their contribution to revenue.

**Champions generated 39.06% of total revenue**, making them the highest revenue-generating customer segment.

The next major contributors were:

- At Risk High Value — **16.20%**
- Regular Customers — **13.58%**
- Loyal Customers — **13.35%**

This helped identify which customer groups are most important from a revenue perspective and which high-value customers may need retention efforts.

### Repeat Purchase Analysis

Customer purchase frequency was analyzed to understand repeat purchasing behavior.

- **99,084 customers (95.4%)** were repeat customers.
- **11,434 customers (4.6%)** were one-time customers.

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

This helped connect **customer value with product preferences** and provides a basis for more targeted customer strategies.

## Key Insights

- **Champions contribute 39.06% of total revenue** while representing 14.90% of customers.
- **95.4% of customers are repeat customers**, showing strong repeat purchasing behavior in the dataset.
- **At Risk High Value customers contribute 16.20% of revenue**, making them an important group for customer retention.
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
├── .gitignore
└── README.md
```

The raw dataset and generated CSV files are excluded from the GitHub repository using `.gitignore` because of their size.

## Outcome

This project helped me work through an end-to-end customer analytics workflow — from exploring and processing raw e-commerce event data in Python to performing customer segmentation and retention analysis, and finally presenting the results through Power BI.

It combines **customer analytics, business analysis, data visualization, and Python-based data processing** in one project.