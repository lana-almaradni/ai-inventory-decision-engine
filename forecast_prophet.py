"""
forecast_prophet.py
-------------------
AI-Powered Inventory & Sales Decision Engine
Author: Lana Al Maradni

What this script does:
1. Loads monthly sales time series data exported from Power BI
2. Runs Meta's Prophet forecasting model for each product individually
3. Generates a 3-month demand forecast with 95% confidence intervals
4. Outputs results as CSV to be loaded back into Power BI

Input  : data/sales_timeseries.csv
Output : data/product_forecast.csv

Notes:
- Products with fewer than 6 months of history are skipped
- yhat       = forecasted quantity (best estimate)
- yhat_lower = lower bound of 95% confidence interval
- yhat_upper = upper bound of 95% confidence interval
- A wide confidence interval indicates unpredictable demand (Z products)
- A narrow confidence interval indicates stable demand (X products)
"""

import pandas as pd
from prophet import Prophet

# ========================
# LOAD DATA
# ========================
INPUT_PATH  = "data/sales_timeseries.csv"
OUTPUT_PATH = "data/product_forecast.csv"

df = pd.read_csv(INPUT_PATH)

# ========================
# CLEAN & RENAME COLUMNS
# ========================
df.columns = [c.strip() for c in df.columns]

# Prophet requires exactly 'ds' (date) and 'y' (value)
df = df.rename(columns={
    "InvoiceDate":    "ds",
    "Sum of Quantity": "y"
})

df["ds"] = pd.to_datetime(df["ds"])

# ========================
# FORECAST PER PRODUCT
# ========================
products      = df["ProductID"].unique()
forecast_list = []

for product in products:

    product_df = df[df["ProductID"] == product][["ds", "y"]].copy()

    # Skip products with insufficient history
    # Prophet needs at least 6 data points to detect meaningful patterns
    if len(product_df) < 6:
        continue

    # Fit Prophet model
    # interval_width=0.95 produces 95% confidence interval bounds
    model = Prophet(interval_width=0.95)
    model.fit(product_df)

    # Generate future dates — 3 months beyond last data point
    future   = model.make_future_dataframe(periods=3, freq="ME")
    forecast = model.predict(future)

    forecast["ProductID"] = product
    forecast_list.append(
        forecast[["ds", "ProductID", "yhat", "yhat_lower", "yhat_upper"]]
    )

# ========================
# COMBINE & CLEAN OUTPUT
# ========================
final_forecast = pd.concat(forecast_list)
final_forecast = final_forecast.sort_values(["ProductID", "ds"]).reset_index(drop=True)
final_forecast[["yhat", "yhat_lower", "yhat_upper"]] = (
    final_forecast[["yhat", "yhat_lower", "yhat_upper"]].round(2)
)

# ========================
# SAVE OUTPUT
# ========================
final_forecast.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"Done: {OUTPUT_PATH} created")
print(f"Total rows : {len(final_forecast)}")
print(f"Products   : {final_forecast['ProductID'].nunique()}")
print(f"Columns    : {final_forecast.columns.tolist()}")
