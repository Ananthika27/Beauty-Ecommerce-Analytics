-- Example SQL for the cleaned purchase data (one row per purchased item).
-- Assumes a table `purchases(event_time, user_id, user_session, product_id,
-- category_id, brand, price)` loaded from the cleaned events. Standard SQL.

-- 1. Monthly revenue, orders and customers
SELECT strftime(event_time, '%Y-%m')   AS month,
       COUNT(DISTINCT user_id)         AS purchasing_customers,
       COUNT(DISTINCT user_session)    AS orders,
       SUM(price)                      AS revenue,
       SUM(price) / COUNT(DISTINCT user_session) AS revenue_per_order
FROM purchases
GROUP BY 1
ORDER BY 1;

-- 2. One-time vs repeat customers (an order = a distinct session)
WITH per_user AS (
    SELECT user_id, COUNT(DISTINCT user_session) AS orders, SUM(price) AS spent
    FROM purchases
    GROUP BY user_id
)
SELECT CASE WHEN orders = 1 THEN 'One-Time' ELSE 'Repeat' END AS customer_type,
       COUNT(*)                                 AS customers,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_customers,
       ROUND(100.0 * SUM(spent) / SUM(SUM(spent)) OVER (), 2) AS pct_revenue
FROM per_user
GROUP BY 1;

-- 3. Top 10 brands by revenue
SELECT brand, COUNT(*) AS items_sold, ROUND(SUM(price), 2) AS revenue
FROM purchases
WHERE brand IS NOT NULL
GROUP BY brand
ORDER BY revenue DESC
LIMIT 10;

-- 4. Cohort retention: customers returning N months after first purchase
WITH monthly AS (
    SELECT DISTINCT user_id, DATE_TRUNC('month', event_time) AS m FROM purchases
),
cohorts AS (
    SELECT user_id, MIN(m) AS cohort FROM monthly GROUP BY user_id
)
SELECT strftime(c.cohort, '%Y-%m') AS cohort,
       DATE_DIFF('month', c.cohort, m.m) + 1 AS period,
       COUNT(DISTINCT m.user_id) AS customers
FROM monthly m JOIN cohorts c USING (user_id)
GROUP BY 1, 2
ORDER BY 1, 2;

-- 5. RFM base table
SELECT user_id,
       DATE_DIFF('day', MAX(event_time), (SELECT MAX(event_time) FROM purchases) + INTERVAL 1 DAY) AS recency,
       COUNT(DISTINCT user_session) AS frequency,
       SUM(price) AS monetary
FROM purchases
GROUP BY user_id;
