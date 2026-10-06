import pandas as pd
from pathlib import Path

data_folder = Path("data")

files = list(data_folder.glob("*.csv"))

dataframes = []

for file in files:
    df = pd.read_csv(file)
    dataframes.append(df)

df = pd.concat(dataframes, ignore_index=True)

print("Combined dataset:")
print("Rows:", f"{len(df):,}")
print("Columns:", len(df.columns))
print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())
df["event_time"] = pd.to_datetime(df["event_time"], utc=True)

print("\nEvent time data type:")
print(df["event_time"].dtype)
df["event_time"] = pd.to_datetime(df["event_time"], utc=True)

print("\nEvent time data type:")
print(df["event_time"].dtype)
print("\nDate range:")
print("Start:", df["event_time"].min())
print("End:", df["event_time"].max())
print("\nEvent types:")
print(df["event_type"].value_counts())

print("\nEvent type percentages:")
print(df["event_type"].value_counts(normalize=True).mul(100).round(2))
print("\nUnique values:")

print("Unique users:", df["user_id"].nunique())
print("Unique sessions:", df["user_session"].nunique())
print("Unique products:", df["product_id"].nunique())
print("Unique categories:", df["category_id"].nunique())
print("Unique brands:", df["brand"].nunique())
print("\nMissing value percentage:")

missing_percent = df.isnull().mean() * 100
print(missing_percent.round(2))
print("\nDuplicate rows:")
print(df.duplicated().sum())
print("\nDuplicate percentage:")
print(round(df.duplicated().mean() * 100, 2), "%")
print("\nRemoving duplicate rows...")

df = df.drop_duplicates()

print("Rows after removing duplicates:", f"{len(df):,}")
print("Duplicates remaining:", df.duplicated().sum())
print("\nPrice check:")

print("Minimum price:", df["price"].min())
print("Maximum price:", df["price"].max())
print("Zero or negative prices:", (df["price"] <= 0).sum())
print("\nNegative prices:", (df["price"] < 0).sum())
print("Zero prices:", (df["price"] == 0).sum())
print("\nZero/negative price events:")
print(
    df[df["price"] <= 0]["event_type"].value_counts()
)
print("\nRemoving invalid prices...")

df = df[df["price"] > 0].copy()

print("Rows after removing invalid prices:", f"{len(df):,}")
print("Invalid prices remaining:", (df["price"] <= 0).sum())
print("\nMissing user_session by event type:")
print(
    df[df["user_session"].isna()]["event_type"].value_counts()
)
print("\nRemoving rows with missing user_session...")

df = df.dropna(subset=["user_session"]).copy()

print("Rows after removing missing sessions:", f"{len(df):,}")
print("Missing user_session remaining:", df["user_session"].isna().sum())
print("\nDate range after cleaning:")
print("Start:", df["event_time"].min())
print("End:", df["event_time"].max())
print("\nEvent types after cleaning:")
print(df["event_type"].value_counts())

print("\nEvent type percentages after cleaning:")
print(
    df["event_type"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)
print("\nPurchase analysis:")

purchases = df[df["event_type"] == "purchase"].copy()

print("Purchase events:", f"{len(purchases):,}")
print("Total purchase value:", round(purchases["price"].sum(), 2))
print("Average purchase price:", round(purchases["price"].mean(), 2))
print("\nMonthly sales:")

purchases["month"] = purchases["event_time"].dt.to_period("M")

monthly_sales = purchases.groupby("month").agg(
    purchase_events=("price", "count"),
    total_sales=("price", "sum"),
    average_price=("price", "mean")
)

print(monthly_sales.round(2))
print("\nMonthly event trends:")

monthly_events = pd.crosstab(
    df["event_time"].dt.to_period("M"),
    df["event_type"]
)

print(monthly_events)
print("\nUser funnel:")

user_event_counts = df.groupby("event_type")["user_id"].nunique()

print(user_event_counts)
print("\nUser conversion rates:")

view_users = user_event_counts["view"]
cart_users = user_event_counts["cart"]
purchase_users = user_event_counts["purchase"]

print("View → Cart:", round(cart_users / view_users * 100, 2), "%")
print("Cart → Purchase:", round(purchase_users / cart_users * 100, 2), "%")
print("View → Purchase:", round(purchase_users / view_users * 100, 2), "%")
print("\nTop 15 products by purchase revenue:")

top_products = (
    purchases.groupby("product_id")
    .agg(
        purchase_events=("price", "count"),
        total_sales=("price", "sum"),
        average_price=("price", "mean")
    )
    .sort_values("total_sales", ascending=False)
    .head(15)
)

print(top_products.round(2))
print("\nTop 15 products by purchase volume:")

top_products_volume = (
    purchases.groupby("product_id")
    .size()
    .sort_values(ascending=False)
    .head(15)
)

print(top_products_volume)
print("\nTop 15 brands by purchase revenue:")

brand_sales = (
    purchases.groupby("brand")
    .agg(
        purchase_events=("price", "count"),
        total_sales=("price", "sum"),
        average_price=("price", "mean")
    )
    .sort_values("total_sales", ascending=False)
    .head(15)
)

print(brand_sales.round(2))
print("\nBrand conversion performance:")

brand_funnel = (
    df.groupby(["brand", "event_type"])
    .size()
    .unstack(fill_value=0)
)

brand_funnel["view_to_purchase_rate"] = (
    brand_funnel["purchase"] /
    brand_funnel["view"] * 100
)

brand_funnel = (
    brand_funnel[
        brand_funnel["view"] >= 1000
    ]
    .sort_values("view_to_purchase_rate", ascending=False)
    .head(15)
)

print(brand_funnel.round(2))
print("\nTop 15 categories by purchase revenue:")

category_sales = (
    purchases.groupby("category_id")
    .agg(
        purchase_events=("price", "count"),
        total_sales=("price", "sum"),
        average_price=("price", "mean")
    )
    .sort_values("total_sales", ascending=False)
    .head(15)
)

print(category_sales.round(2))
print("\nTop 15 categories by purchase volume:")

category_volume = (
    purchases.groupby("category_id")
    .size()
    .sort_values(ascending=False)
    .head(15)
)

print(category_volume)
print("\nCart abandonment analysis:")

cart_users = set(df.loc[df["event_type"] == "cart", "user_id"])
purchase_users = set(df.loc[df["event_type"] == "purchase", "user_id"])

abandoned_users = cart_users - purchase_users

print("Users who added to cart:", len(cart_users))
print("Users who purchased:", len(purchase_users))
print("Users who abandoned cart:", len(abandoned_users))

abandonment_rate = len(abandoned_users) / len(cart_users) * 100

print("Cart abandonment rate:", round(abandonment_rate, 2), "%")
print("\nPurchase revenue by day of week:")

purchases["day_of_week"] = purchases["event_time"].dt.day_name()

weekday_sales = (
    purchases.groupby("day_of_week")["price"]
    .agg(["count", "sum", "mean"])
    .sort_values("sum", ascending=False)
)

weekday_sales.columns = [
    "purchase_events",
    "total_sales",
    "average_price"
]

print(weekday_sales)
print("\nPurchase revenue by hour:")

purchases["hour"] = purchases["event_time"].dt.hour

hourly_sales = (
    purchases.groupby("hour")["price"]
    .agg(["count", "sum", "mean"])
    .sort_values("sum", ascending=False)
)

hourly_sales.columns = [
    "purchase_events",
    "total_sales",
    "average_price"
]

print(hourly_sales)
print("\nCustomer purchase analysis:")

customer_analysis = (
    purchases.groupby("user_id")["price"]
    .agg(["count", "sum", "mean"])
    .sort_values("sum", ascending=False)
)

customer_analysis.columns = [
    "purchase_count",
    "total_spent",
    "average_purchase_value"
]

print("\nTop 15 customers by total spending:")
print(customer_analysis.head(15))

print("\nCustomer spending statistics:")
print(customer_analysis["total_spent"].describe())

# ============================================================
# RFM CUSTOMER SEGMENTATION
# ============================================================

print("\n" + "=" * 60)
print("RFM CUSTOMER SEGMENTATION")
print("=" * 60)

# Use only purchase events
purchases = df[df["event_type"] == "purchase"].copy()

# Reference date = day after the last purchase
reference_date = purchases["event_time"].max() + pd.Timedelta(days=1)

# Calculate RFM metrics
rfm = purchases.groupby("user_id").agg(
    recency=("event_time", lambda x: (reference_date - x.max()).days),
    frequency=("event_time", "count"),
    monetary=("price", "sum")
)

print("\nRFM customer data:")
print(rfm.head())

print("\nRFM summary:")
print(rfm.describe())

# Create RFM scores
rfm["recency_score"] = pd.qcut(
    rfm["recency"],
    5,
    labels=[5, 4, 3, 2, 1],
    duplicates="drop"
).astype(int)

rfm["frequency_score"] = pd.qcut(
    rfm["frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5],
    duplicates="drop"
).astype(int)

rfm["monetary_score"] = pd.qcut(
    rfm["monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5],
    duplicates="drop"
).astype(int)

# Combined RFM score
rfm["rfm_score"] = (
    rfm["recency_score"].astype(str)
    + rfm["frequency_score"].astype(str)
    + rfm["monetary_score"].astype(str)
)

print("\nRFM score distribution:")
print(rfm["rfm_score"].value_counts().head(20))

print("\nTop customers by RFM score:")
print(
    rfm.sort_values(
        ["recency_score", "frequency_score", "monetary_score"],
        ascending=False
    ).head(15)
)

# ============================================================
# RFM CUSTOMER SEGMENTS
# ============================================================

print("\n" + "=" * 60)
print("RFM CUSTOMER SEGMENTS")
print("=" * 60)

def assign_segment(row):
    r = row["recency_score"]
    f = row["frequency_score"]
    m = row["monetary_score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"

    elif r >= 3 and f >= 4:
        return "Loyal Customers"

    elif r >= 4 and f <= 2 and m >= 3:
        return "Potential Loyalists"

    elif r >= 4 and f <= 2:
        return "New Customers"

    elif r <= 2 and f >= 4 and m >= 4:
        return "At Risk High Value"

    elif r <= 2 and f >= 3:
        return "At Risk"

    elif r <= 2 and f <= 2 and m >= 3:
        return "High Value Lost"

    else:
        return "Regular Customers"


rfm["segment"] = rfm.apply(assign_segment, axis=1)

print("\nCustomer segment counts:")
print(rfm["segment"].value_counts())

print("\nCustomer segment percentages:")
print(
    (rfm["segment"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nSegment summary:")

segment_summary = rfm.groupby("segment").agg(
    customers=("segment", "count"),
    avg_recency=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean"),
    total_revenue=("monetary", "sum")
).sort_values("total_revenue", ascending=False)

print(segment_summary.round(2))

# ============================================================
# SEGMENT REVENUE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("SEGMENT REVENUE ANALYSIS")
print("=" * 60)

segment_revenue = rfm.groupby("segment").agg(
    customers=("monetary", "count"),
    total_revenue=("monetary", "sum"),
    average_customer_value=("monetary", "mean"),
    average_frequency=("frequency", "mean"),
    average_recency=("recency", "mean")
).sort_values("total_revenue", ascending=False)

segment_revenue["revenue_percentage"] = (
    segment_revenue["total_revenue"]
    / segment_revenue["total_revenue"].sum()
    * 100
)

print("\nRevenue by customer segment:")
print(segment_revenue.round(2))

print("\nTop revenue-generating segments:")
print(
    segment_revenue[
        ["customers", "total_revenue", "revenue_percentage"]
    ].head(5).round(2)
)

# ============================================================
# CUSTOMER SEGMENT PRODUCT ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER SEGMENT PRODUCT ANALYSIS")
print("=" * 60)

# Get purchase events with customer segment
segment_purchases = purchases.merge(
    rfm[["segment"]],
    left_on="user_id",
    right_index=True,
    how="inner"
)

# ------------------------------------------------------------
# Revenue by customer segment
# ------------------------------------------------------------

segment_product_revenue = (
    segment_purchases
    .groupby(["segment", "product_id"])
    .agg(
        purchase_events=("product_id", "count"),
        total_sales=("price", "sum")
    )
    .reset_index()
)

print("\nTop 5 products by revenue for each customer segment:")

for segment in segment_product_revenue["segment"].unique():

    print(f"\n--- {segment} ---")

    top_products = (
        segment_product_revenue[
            segment_product_revenue["segment"] == segment
        ]
        .sort_values("total_sales", ascending=False)
        .head(5)
    )

    print(top_products)


# ------------------------------------------------------------
# Brand preference by customer segment
# ------------------------------------------------------------

segment_brand_revenue = (
    segment_purchases
    .groupby(["segment", "brand"])
    .agg(
        purchase_events=("brand", "count"),
        total_sales=("price", "sum")
    )
    .reset_index()
)

print("\nTop 5 brands by revenue for each customer segment:")

for segment in segment_brand_revenue["segment"].unique():

    print(f"\n--- {segment} ---")

    top_brands = (
        segment_brand_revenue[
            segment_brand_revenue["segment"] == segment
        ]
        .sort_values("total_sales", ascending=False)
        .head(5)
    )

    print(top_brands)
    
    # ============================================================
# CUSTOMER SEGMENT CATEGORY ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER SEGMENT CATEGORY ANALYSIS")
print("=" * 60)

segment_category_revenue = (
    segment_purchases
    .groupby(["segment", "category_id"])
    .agg(
        purchase_events=("category_id", "count"),
        total_sales=("price", "sum")
    )
    .reset_index()
)

print("\nTop 5 categories by revenue for each customer segment:")

for segment in segment_category_revenue["segment"].unique():

    print(f"\n--- {segment} ---")

    top_categories = (
        segment_category_revenue[
            segment_category_revenue["segment"] == segment
        ]
        .sort_values("total_sales", ascending=False)
        .head(5)
    )

    print(top_categories)
    
    # ============================================================
# CUSTOMER REPEAT PURCHASE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER REPEAT PURCHASE ANALYSIS")
print("=" * 60)

# Count purchases made by each customer
customer_purchase_frequency = (
    purchases
    .groupby("user_id")
    .agg(
        purchase_count=("user_id", "count"),
        total_spent=("price", "sum")
    )
)

# ------------------------------------------------------------
# Classify customers
# ------------------------------------------------------------

customer_purchase_frequency["customer_type"] = (
    customer_purchase_frequency["purchase_count"]
    .apply(
        lambda x: "One-Time Customer"
        if x == 1
        else "Repeat Customer"
    )
)

# ------------------------------------------------------------
# Overall repeat purchase statistics
# ------------------------------------------------------------

customer_type_summary = (
    customer_purchase_frequency
    .groupby("customer_type")
    .agg(
        customers=("purchase_count", "count"),
        total_revenue=("total_spent", "sum"),
        average_spending=("total_spent", "mean"),
        average_purchases=("purchase_count", "mean")
    )
)

customer_type_summary["customer_percentage"] = (
    customer_type_summary["customers"]
    / customer_type_summary["customers"].sum()
    * 100
)

customer_type_summary["revenue_percentage"] = (
    customer_type_summary["total_revenue"]
    / customer_type_summary["total_revenue"].sum()
    * 100
)

print("\nOne-Time vs Repeat Customers:")
print(customer_type_summary.round(2))

# ------------------------------------------------------------
# Purchase frequency distribution
# ------------------------------------------------------------

print("\nPurchase frequency distribution:")

purchase_frequency_distribution = (
    customer_purchase_frequency["purchase_count"]
    .value_counts()
    .sort_index()
)

print(purchase_frequency_distribution.head(20))

# ============================================================
# MONTHLY CUSTOMER BEHAVIOR ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("MONTHLY CUSTOMER BEHAVIOR ANALYSIS")
print("=" * 60)

# Create month column
purchases["month"] = purchases["event_time"].dt.to_period("M")

# Monthly customer and revenue metrics
monthly_customer_analysis = (
    purchases
    .groupby("month")
    .agg(
        purchasing_customers=("user_id", "nunique"),
        purchase_events=("user_id", "count"),
        total_revenue=("price", "sum"),
        average_purchase_value=("price", "mean")
    )
    .reset_index()
)

print("\nMonthly customer behavior:")
print(monthly_customer_analysis.round(2))

# ============================================================
# CUSTOMER COHORT RETENTION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER COHORT RETENTION ANALYSIS")
print("=" * 60)

# Work with purchase events only
cohort_data = purchases[["user_id", "event_time"]].copy()

# Month of each purchase
cohort_data["purchase_month"] = (
    cohort_data["event_time"].dt.to_period("M")
)

# First purchase month for each customer
cohort_data["cohort_month"] = (
    cohort_data
    .groupby("user_id")["purchase_month"]
    .transform("min")
)

# Calculate number of months since first purchase
cohort_data["cohort_index"] = (
    cohort_data["purchase_month"].astype(int)
    - cohort_data["cohort_month"].astype(int)
    + 1
)

# Count unique customers in each cohort/month
cohort_table = (
    cohort_data
    .groupby(["cohort_month", "cohort_index"])["user_id"]
    .nunique()
    .unstack(fill_value=0)
)

# Convert counts into retention percentages
cohort_sizes = cohort_table.iloc[:, 0]

retention_table = (
    cohort_table
    .divide(cohort_sizes, axis=0)
    * 100
)

print("\nCohort customer counts:")
print(cohort_table)

print("\nCohort retention percentages:")
print(retention_table.round(2))

# ============================================================
# EXPORT DASHBOARD DATA
# ============================================================

monthly_customer_analysis.to_csv(
    "monthly_customer_behavior.csv",
    index=False
)

print("\nMonthly customer behavior CSV created successfully!")

# ============================================================
# EXPORT RFM CUSTOMER SEGMENTATION
# ============================================================

rfm.reset_index().to_csv(
    "rfm_customer_segments.csv",
    index=False
)

print("\nRFM customer segments CSV created successfully!")

# ============================================================
# EXPORT CUSTOMER SEGMENT REVENUE ANALYSIS
# ============================================================

segment_revenue.reset_index().to_csv(
    "segment_revenue_analysis.csv",
    index=False
)

print("\nSegment revenue analysis CSV created successfully!")

# ============================================================
# EXPORT CUSTOMER SEGMENT PRODUCT ANALYSIS
# ============================================================

segment_product_revenue.to_csv(
    "segment_product_analysis.csv",
    index=False
)

print("\nSegment product analysis CSV created successfully!")
# ============================================================
# EXPORT CUSTOMER SEGMENT BRAND ANALYSIS
# ============================================================

segment_brand_revenue.to_csv(
    "segment_brand_analysis.csv",
    index=False
)

print("\nSegment brand analysis CSV created successfully!")
# ============================================================
# EXPORT CUSTOMER SEGMENT CATEGORY ANALYSIS
# ============================================================

segment_category_revenue.to_csv(
    "segment_category_analysis.csv",
    index=False
)

print("\nSegment category analysis CSV created successfully!")
# ============================================================
# EXPORT CUSTOMER REPEAT PURCHASE ANALYSIS
# ============================================================

customer_purchase_frequency.to_csv(
    "repeat_purchase_analysis.csv",
    index=False
)

print("\nRepeat purchase analysis CSV created successfully!")