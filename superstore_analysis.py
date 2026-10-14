"""
Project 1: Superstore Sales Analytics, KPI Reporting & Visual EDA
Author: Vallimpula Niveditha, IIT Jodhpur
Tools: Python (Pandas, NumPy, Matplotlib, Seaborn)
Description: Complete end-to-end analytics script that calculates executive KPIs,
             performs deep-dive Exploratory Data Analysis (EDA), plots 4 professional
             charts, exports cleaned data for Power BI/Excel, and saves the chart image.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_complete_sales_analytics():
    print("=" * 65)
    print("PROJECT 1: SUPERSTORE SALES ANALYTICS & VISUAL EDA")
    print("Author: Vallimpula Niveditha | IIT Jodhpur")
    print("=" * 65)

    # -------------------------------------------------------------
    # 1. Simulate & Prepare Transactional Dataset (5,000 Orders)
    # -------------------------------------------------------------
    np.random.seed(42)
    n_records = 5000

    categories = ['Technology', 'Furniture', 'Office Supplies']
    sub_cats = {
        'Technology': ['Phones', 'Accessories', 'Machines', 'Copiers'],
        'Furniture': ['Chairs', 'Tables', 'Bookcases', 'Furnishings'],
        'Office Supplies': ['Storage', 'Binders', 'Paper', 'Art', 'Envelopes']
    }
    regions = ['North', 'South', 'East', 'West']

    cat_choice = np.random.choice(categories, n_records, p=[0.30, 0.30, 0.40])
    subcat_choice = [np.random.choice(sub_cats[c]) for c in cat_choice]

    df = pd.DataFrame({
        'Order_ID': [f'ORD-{10000 + i}' for i in range(n_records)],
        'Customer_ID': [f'CUST-{np.random.randint(100, 800)}' for _ in range(n_records)],
        'Order_Date': pd.date_range(start='2023-01-01', periods=n_records, freq='h'),
        'Region': np.random.choice(regions, n_records),
        'Category': cat_choice,
        'Sub_Category': subcat_choice,
        'Sales': np.round(np.random.uniform(20.0, 1500.0, n_records), 2),
        'Discount': np.random.choice([0.0, 0.1, 0.2, 0.3, 0.4], n_records, p=[0.40, 0.25, 0.15, 0.12, 0.08])
    })

    # Realistic profit formula: base margin minus discount penalty
    margin_mult = {'Technology': 0.28, 'Office Supplies': 0.20, 'Furniture': 0.12}
    df['Base_Margin'] = df['Category'].map(margin_mult)
    # Discounts over 20% severely erode profit and cause negative margins
    df['Profit'] = np.round(df['Sales'] * (df['Base_Margin'] - (1.1 * df['Discount'])), 2)
    df.drop(columns=['Base_Margin'], inplace=True)

    # -------------------------------------------------------------
    # 2. Executive KPIs Calculation
    # -------------------------------------------------------------
    total_revenue = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    avg_margin = (total_profit / total_revenue) * 100

    print(f"\n[EXECUTIVE KPI SUMMARY]")
    print(f"Total Transactions  : {len(df):,}")
    print(f"Unique Customers    : {df['Customer_ID'].nunique():,}")
    print(f"Total Gross Revenue : INR {total_revenue:,.2f}")
    print(f"Total Net Profit    : INR {total_profit:,.2f}")
    print(f"Overall Profit Margin: {avg_margin:.2f}%")

    # Category Breakdown Table
    print("\n[PERFORMANCE BY CATEGORY]")
    cat_summary = df.groupby('Category').agg(
        Total_Sales=('Sales', 'sum'),
        Total_Profit=('Profit', 'sum'),
        Order_Count=('Order_ID', 'count')
    ).reset_index()
    cat_summary['Profit_Margin_%'] = (cat_summary['Total_Profit'] / cat_summary['Total_Sales']) * 100
    print(cat_summary.to_string(index=False))

    # -------------------------------------------------------------
    # 3. Professional Visual EDA (4-Chart Executive Dashboard)
    # -------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Superstore Performance & Exploratory Data Analysis', fontsize=16, fontweight='bold')

    # Chart 1: Sales vs. Profit by Category
    cat_plot = df.groupby('Category')[['Sales', 'Profit']].sum()
    cat_plot.plot(kind='bar', ax=axes[0, 0], color=['#38bdf8', '#0284c7'], edgecolor='black')
    axes[0, 0].set_title('1. Sales vs. Profit by Category', fontsize=13, fontweight='bold')
    axes[0, 0].set_ylabel('INR')
    axes[0, 0].tick_params(axis='x', rotation=0)

    # Chart 2: The Discount Trap (Profit vs. Discount)
    sns.lineplot(data=df, x='Discount', y='Profit', ax=axes[0, 1], color='#ef4444', linewidth=2.5, marker='o')
    axes[0, 1].axhline(0, color='black', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('2. The Discount Trap: Profit Erosion', fontsize=13, fontweight='bold')
    axes[0, 1].set_ylabel('Average Profit (INR)')

    # Chart 3: Regional Profitability Heatmap
    region_pivot = df.pivot_table(index='Region', columns='Category', values='Profit', aggfunc='sum')
    sns.heatmap(region_pivot, annot=True, fmt='.0f', cmap='Blues', ax=axes[1, 0], cbar=False)
    axes[1, 0].set_title('3. Regional Profit Distribution (Heatmap)', fontsize=13, fontweight='bold')

    # Chart 4: Top Sub-Categories by Profit
    subcat_profit = df.groupby('Sub_Category')['Profit'].sum().sort_values(ascending=False).head(8)
    subcat_profit.plot(kind='barh', ax=axes[1, 1], color='#10b981', edgecolor='black')
    axes[1, 1].set_title('4. Top Profitable Sub-Categories', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Profit (INR)')

    plt.tight_layout()

    # Save visual dashboard as image for PPT slides
    plt.savefig('superstore_dashboard_charts.png', dpi=300)
    plt.show()

    # -------------------------------------------------------------
    # 4. Actionable Business Insights & Export
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("KEY BUSINESS TAKEAWAYS FOR INTERVIEWS / REPORT:")
    print("=" * 65)
    print("1. DISCOUNT THRESHOLD: Discounts above 20% cause average profit to plunge below ZERO.")
    print("2. CATEGORY DRIVERS: Technology produces highest margin due to premium hardware pricing.")
    print("3. RECOMMENDATION: Cap Furniture discounts at 15% to stop margin leakage.")

    # Export cleaned dataset
    df.to_csv('superstore_cleaned_data.csv', index=False)
    print("\n[SUCCESS] Files created in your working directory:")
    print("  -> 'superstore_cleaned_data.csv' (Data file for Excel/Power BI)")
    print("  -> 'superstore_dashboard_charts.png' (High-resolution image for your PPT)")

if __name__ == '__main__':
    run_complete_sales_analytics()
