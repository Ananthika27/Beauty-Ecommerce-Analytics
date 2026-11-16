# Power BI build guide (about 30–40 minutes)

Power BI report files (`.pbix`) are binary and can only be created inside Power BI, so this folder gives you everything needed to build the report quickly and consistently:

| File | Purpose |
| --- | --- |
| `../outputs/powerbi/*.csv` | Clean star-schema data (generate with `make powerbi`) |
| `measures.dax` | All DAX measures, ready to paste |
| `theme.json` | Colour theme (View > Themes > Browse for themes) |
| this guide | Model, relationships and page-by-page layout |

## 0. Which Power BI?

- **Windows:** install free *Power BI Desktop* (Microsoft Store). Best option.
- **Mac:** Desktop doesn't run on macOS. Use **app.powerbi.com** (free account with a work/school email): *Create > Upload a file* and upload the CSVs, then build visuals in the browser. Or run Desktop in a Windows VM or on a Windows PC.

## 1. Generate and load the data

```bash
make all          # if you haven't already
make powerbi      # writes outputs/powerbi/*.csv
```

In Desktop: **Get Data > Text/CSV**, load each file in `outputs/powerbi/` (or *Get Data > Folder* then split). 16 files, all small except `fact_customers.csv` (110,518 rows). Use **Load**, not Transform; the data is already clean.

## 2. Model (Model view)

Create these relationships (many-to-one, single direction):

| From (many) | To (one) |
| --- | --- |
| `fact_customers[segment]` | `dim_segment[segment]` |
| `fact_segment_brand_top10[segment]` | `dim_segment[segment]` |

Other tables stand alone. Then set **Sort by column**:
- `dim_segment[segment]` by `sort_order`
- `fact_weekday[day_of_week]` by `day_number`
- `fact_funnel[stage]`: sort by `users` descending in the visual

Import `theme.json` via **View > Themes > Browse for themes**. Create the DAX measures from `measures.dax` (**Modeling > New measure**, one at a time).

## 3. Pages

### Page 1: Overview
- **Cards:** `Customers`, `Revenue`, `Repeat Rate`, `Repeat Revenue Share`
- **Donut:** Legend `fact_repeat_summary[customer_type]`, Values `customer_percentage`
- **Clustered bar:** Axis `dim_segment[segment]`, Values `Customers % of Total` and `Revenue % of Total`
- **Title text box:** "78.9% of customers buy once; repeat buyers bring 48% of revenue"

### Page 2: RFM segments
- **Slicer:** `dim_segment[segment]`
- **Bar:** segment vs `Revenue`
- **Table:** segment, `Customers`, `Avg Spend per Customer`, `Avg Orders per Customer`, `Avg Days Since Last Order`, `recommended_action`
- **Scatter:** X `Avg Days Since Last Order`, Y `Avg Spend per Customer`, size `Customers`, details `segment`

### Page 3: Monthly trend and timing
- **Line + columns:** Axis `fact_monthly[month]`, columns `Monthly Revenue`, line `Revenue per Order`
- **Card:** `Revenue MoM %`
- **Column:** `fact_weekday[day_of_week]` vs `total_sales`
- **Line:** `fact_hour[hour_utc]` vs `total_sales`

### Page 4: Cohort retention and funnel
- **Matrix:** Rows `fact_cohort[cohort]`, Columns `months_since_first_purchase`, Values `Avg Retention %`; add *conditional formatting > background colour scale* for a heatmap
- **Funnel visual:** Category `fact_funnel[stage]`, Values `Funnel Users`
- **Table:** `fact_brand_funnel` sorted by `cart_to_purchase_pct` (add a Top N filter of 15)

### Page 5: Brands and products
- **Slicer:** `dim_segment[segment]`
- **Bar (Top N = 10):** `fact_segment_brand_top10[brand]` vs `Brand Sales`
- **Table:** `fact_brand_affinity` (`brand_a`, `brand_b`, `orders_together`, `lift`), sorted by `lift`

### Page 6: Retention model
- **Table:** `fact_model_metrics` (model, roc_auc, roc_auc_ci_low, roc_auc_ci_high, top_decile_lift)
- **Column + line:** `fact_model_deciles[decile]`: columns `actual_rate`, line `avg_score`
- **Cards:** `Win-Back Customers`, `Win-Back Historic Spend`
- **Table (target list):** `fact_customers` fields `user_id`, `segment`, `purchase_probability`, `total_spent`, `orders`; filter `probability_band` and segment. Add an **Export data** option so a marketing team can download it.
- **Table (why they won't return):** `fact_winback_reasons` (`user_id`, `segment`, `purchase_probability`, `main_reason_1..3`)
- **Column chart:** `fact_clv_validation[decile]` with `predicted_value` and `actual_value`, plus cards `Predicted 90d Value` and `Avg Probability Alive`
- **Note text box:** "Scores are for *ranking* customers. The monthly return rate fell from 18.5% to 8.5% as the customer base grew."

## 4. Finish

1. Set page size 16:9, align visuals to a grid, same card height across pages.
2. Add a short text box on each page with the one-line takeaway.
3. Save as `dashboard/Beauty_Ecommerce_Analytics.pbix`.
4. Export screenshots: *File > Export > PDF*, or screenshot each page into `images/powerbi_*.png` and embed them in the README.

## Notes
- Amounts are in the dataset's own price units (no currency stated).
- Revenue is item price summed over purchases, not profit.
- If a visual shows "(Blank)" for segment, check the relationship in step 2.
