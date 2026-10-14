-- ====================================================================
-- PROJECT 1: Superstore Sales Performance & Customer Analytics
-- Author: Vallimpula Niveditha (IIT Jodhpur)
-- Description: Advanced SQL Queries for Business Insights, RFM, and KPIs
-- ====================================================================

-- 1. Create Orders Table
CREATE TABLE IF NOT EXISTS superstore_orders (
    order_id VARCHAR(20),
    order_date DATE,
    customer_id VARCHAR(20),
    customer_name VARCHAR(100),
    segment VARCHAR(50),
    region VARCHAR(50),
    category VARCHAR(50),
    sub_category VARCHAR(50),
    sales NUMERIC(10, 2),
    profit NUMERIC(10, 2),
    discount NUMERIC(4, 2)
);

-- --------------------------------------------------------------------
-- QUERY 1: KPI Summary - Total Revenue, Profit, and Profit Margin %
-- --------------------------------------------------------------------
SELECT 
    COUNT(DISTINCT order_id) AS total_orders,
    COUNT(DISTINCT customer_id) AS total_customers,
    ROUND(SUM(sales), 2) AS total_revenue,
    ROUND(SUM(profit), 2) AS total_profit,
    ROUND((SUM(profit) / SUM(sales)) * 100, 2) AS profit_margin_pct
FROM superstore_orders;

-- --------------------------------------------------------------------
-- QUERY 2: Regional Performance & Profitability Breakdown
-- --------------------------------------------------------------------
SELECT 
    region,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(SUM(profit), 2) AS total_profit,
    ROUND((SUM(profit) / SUM(sales)) * 100, 2) AS profit_margin_pct
FROM superstore_orders
GROUP BY region
ORDER BY total_profit DESC;

-- --------------------------------------------------------------------
-- QUERY 3: Ranking Top 5 Products by Profit in Each Category (Window Function)
-- Demonstrates DENSE_RANK() and PARTITION BY
-- --------------------------------------------------------------------
WITH RankedProducts AS (
    SELECT 
        category,
        sub_category,
        ROUND(SUM(sales), 2) AS total_sales,
        ROUND(SUM(profit), 2) AS total_profit,
        DENSE_RANK() OVER (PARTITION BY category ORDER BY SUM(profit) DESC) AS rank_in_category
    FROM superstore_orders
    GROUP BY category, sub_category
)
SELECT 
    category,
    sub_category,
    total_sales,
    total_profit,
    rank_in_category
FROM RankedProducts
WHERE rank_in_category <= 5
ORDER BY category, rank_in_category;

-- --------------------------------------------------------------------
-- QUERY 4: Month-over-Month (MoM) Sales Growth using LAG()
-- --------------------------------------------------------------------
WITH MonthlySales AS (
    SELECT 
        DATE_TRUNC('month', order_date) AS sales_month,
        ROUND(SUM(sales), 2) AS current_month_sales
    FROM superstore_orders
    GROUP BY DATE_TRUNC('month', order_date)
)
SELECT 
    sales_month,
    current_month_sales,
    LAG(current_month_sales, 1) OVER (ORDER BY sales_month) AS previous_month_sales,
    ROUND(
        ((current_month_sales - LAG(current_month_sales, 1) OVER (ORDER BY sales_month)) 
        / LAG(current_month_sales, 1) OVER (ORDER BY sales_month)) * 100, 
        2
    ) AS mom_growth_pct
FROM MonthlySales
ORDER BY sales_month;

-- --------------------------------------------------------------------
-- QUERY 5: RFM (Recency, Frequency, Monetary) Customer Segmentation
-- --------------------------------------------------------------------
WITH CustomerStats AS (
    SELECT 
        customer_id,
        customer_name,
        MAX(order_date) AS last_order_date,
        COUNT(DISTINCT order_id) AS total_orders,
        ROUND(SUM(sales), 2) AS total_spend
    FROM superstore_orders
    GROUP BY customer_id, customer_name
)
SELECT 
    customer_id,
    customer_name,
    total_orders AS frequency,
    total_spend AS monetary_value,
    (CURRENT_DATE - last_order_date) AS recency_days,
    NTILE(4) OVER (ORDER BY (CURRENT_DATE - last_order_date) ASC) AS r_score,
    NTILE(4) OVER (ORDER BY total_orders DESC) AS f_score,
    NTILE(4) OVER (ORDER BY total_spend DESC) AS m_score
FROM CustomerStats
ORDER BY total_spend DESC
LIMIT 50;
