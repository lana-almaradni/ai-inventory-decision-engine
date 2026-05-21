"""
xyz_classification.py
---------------------
AI-Powered Inventory & Sales Decision Engine
Author: Lana Al Maradni

What this script does:
1. Loads raw sales data from the Excel dataset
2. Aggregates sales by product and month
3. Calculates the Coefficient of Variation (CV) for both revenue and quantity
4. Classifies each product as X, Y, or Z based on demand predictability
5. Uses the worse of the two CV scores as the final classification
6. Outputs results as CSV to be loaded into Power BI

Input  : data/ai_powered_inventory_sales_engine_dataset.xlsx (sheet: Sales)
Output : data/product_xyz_output.csv

Classification thresholds:
- X → CV < 0.5   : Stable, predictable demand — plan with confidence
- Y → CV <= 1.0  : Moderate variability — plan with seasonal awareness
- Z → CV > 1.0   : Erratic, unpredictable demand — handle with caution

CV = Standard Deviation / Mean (per product, calculated on monthly values)
Combined XYZ uses the worse classification between CV_Revenue and CV_Qty
to ensure the most conservative and risk-aware result.
"""

import pandas as pd
import numpy as np

# ========================
# LOAD DATA
# ========================
INPUT_PATH  = "data/ai_powered_inventory_sales_engine_dataset.xlsx"
OUTPUT_PATH = "data/product_xyz_output.csv"

df = pd.read_excel(INPUT_PATH, sheet_name="Sales")

# ========================
# PREPARE DATA
# ========================
df["OrderDate"] = pd.to_datetime(df["OrderDate"])
df["YearMonth"] = df["OrderDate"].dt.to_period("M")

# ========================
# MONTHLY AGGREGATION
# ========================
monthly_revenue = (
    df.groupby(["ProductID", "YearMonth"])["NetSales"]
    .sum()
    .reset_index()
)

monthly_qty = (
    df.groupby(["ProductID", "YearMonth"])["Quantity"]
    .sum()
    .reset_index()
)

# ========================
# CV CALCULATION
# ========================
# CV = Std / Mean per product across all months
# fillna(0) handles products with only one month of data (std = NaN)

cv_revenue = (
    monthly_revenue.groupby("ProductID")["NetSales"]
    .agg(["mean", "std"])
    .reset_index()
)
cv_revenue.columns = ["ProductID", "Avg_Monthly_Revenue", "Std_Monthly_Revenue"]
cv_revenue["CV_Revenue"] = (
    cv_revenue["Std_Monthly_Revenue"] / cv_revenue["Avg_Monthly_Revenue"]
).fillna(0)

cv_qty = (
    monthly_qty.groupby("ProductID")["Quantity"]
    .agg(["mean", "std"])
    .reset_index()
)
cv_qty.columns = ["ProductID", "Avg_Monthly_Qty", "Std_Monthly_Qty"]
cv_qty["CV_Qty"] = (
    cv_qty["Std_Monthly_Qty"] / cv_qty["Avg_Monthly_Qty"]
).fillna(0)

# ========================
# MERGE
# ========================
xyz_df = cv_revenue.merge(cv_qty, on="ProductID", how="left")

# ========================
# XYZ CLASSIFICATION
# ========================
def classify_xyz(cv):
    """Classify a single CV value into X, Y, or Z."""
    if cv < 0.5:   return "X"
    elif cv <= 1.0: return "Y"
    return "Z"

xyz_df["XYZ_Revenue"] = xyz_df["CV_Revenue"].apply(classify_xyz)
xyz_df["XYZ_Qty"]     = xyz_df["CV_Qty"].apply(classify_xyz)

def combined_xyz(row):
    """
    Return the worse classification between XYZ_Revenue and XYZ_Qty.
    Ensures the final label reflects the highest demand risk.
    Example: XYZ_Revenue=X, XYZ_Qty=Z → Combined=Z
    """
    order = {"X": 0, "Y": 1, "Z": 2}
    worse = max(order[row["XYZ_Revenue"]], order[row["XYZ_Qty"]])
    return ["X", "Y", "Z"][worse]

xyz_df["XYZ_Combined"] = xyz_df.apply(combined_xyz, axis=1)

# ========================
# SAVE OUTPUT
# ========================
output = xyz_df[[
    "ProductID",
    "CV_Revenue",
    "CV_Qty",
    "XYZ_Revenue",
    "XYZ_Qty",
    "XYZ_Combined"
]].round(4)

output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"Done: {OUTPUT_PATH} created")
print(f"Products classified: {len(output)}")
print("\nXYZ Distribution:")
print(output["XYZ_Combined"].value_counts())
print("\nSample output:")
print(output.head(10))
