"""
risk_scoring_clustering.py
--------------------------
AI-Powered Inventory & Sales Decision Engine
Author: Lana Al Maradni

What this script does:
1. Loads product data exported from Power BI as CSV
2. Calculates a risk score (0-100) for every product
3. Classifies risk level: High / Medium / Low
4. Generates an AI recommendation per product
5. Runs KMeans clustering to segment products into 5 behavioral groups
6. Outputs results as CSV to be loaded back into Power BI

Input  : data/product_ai_input.csv
Output : data/product_ai_output_with_clusters.csv
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ========================
# LOAD DATA
# ========================
INPUT_PATH  = "data/product_ai_input.csv"
OUTPUT_PATH = "data/product_ai_output_with_clusters.csv"

df = pd.read_csv(INPUT_PATH)

# ========================
# CLEAN & RENAME COLUMNS
# ========================
df.columns = [c.strip().replace("\n", " ") for c in df.columns]

df = df.rename(columns={
    "Sales Last 12M":        "Sales12M",
    "Sales Last 6M":         "Sales6M",
    "GP % 12M":              "GPPct12M",
    "Months of Cover 6M":    "MonthsCover6M",
    "Avg Stock Age Days":    "AvgStockAgeDays",
    "Current Available Qty": "AvailableQty",
    "Sum of LeadTimeDays":   "LeadTimeDays",
    "Customers Count 12M":   "Customers12M",
    "StrategicFlag":         "StrategicFlag",
    "ABC Sales Class":       "ABCClass",
    "XYZ_Class":             "XYZ_Combined",
    "ProductID":             "ProductID",
    "ProductName":           "ProductName"
})

# ========================
# VALIDATE REQUIRED COLUMNS
# ========================
required_cols = [
    "ProductID", "ProductName",
    "Sales12M", "Sales6M", "AvailableQty",
    "MonthsCover6M", "GPPct12M", "Customers12M",
    "AvgStockAgeDays", "ABCClass", "XYZ_Combined",
    "StrategicFlag", "LeadTimeDays"
]

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in input file: {missing_cols}")

# ========================
# DATA CLEANING
# ========================
numeric_cols = [
    "Sales12M", "Sales6M", "AvailableQty", "MonthsCover6M",
    "GPPct12M", "Customers12M", "AvgStockAgeDays", "LeadTimeDays"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

text_cols = ["ABCClass", "XYZ_Combined", "StrategicFlag", "ProductID", "ProductName"]
for col in text_cols:
    df[col] = df[col].fillna("").astype(str).str.strip()

df["ABCClass"]      = df["ABCClass"].str.upper()
df["XYZ_Combined"]  = df["XYZ_Combined"].str.upper()
df["StrategicFlag"] = df["StrategicFlag"].str.lower()

# Normalize GP% if exported as percentage (e.g. 15 instead of 0.15)
if df["GPPct12M"].max() > 1:
    df["GPPct12M"] = df["GPPct12M"] / 100

# ========================
# ENCODE XYZ CLASS
# ========================
# X = predictable demand → 1
# Y = moderate variability → 2
# Z = erratic demand → 3
# Unknown values default to Y (2)
xyz_map = {"X": 1, "Y": 2, "Z": 3}
df["XYZ_Encoded"] = df["XYZ_Combined"].map(xyz_map).fillna(2).astype(int)

# ========================
# RISK SCORING
# ========================
# Scores range from 0 to 100. Higher = higher inventory risk.
# Each condition contributes points based on business severity.

def calculate_risk(row):
    score = 0

    sales12m      = row["Sales12M"]
    sales6m       = row["Sales6M"]
    available_qty = row["AvailableQty"]
    months_cover  = row["MonthsCover6M"]
    gp_pct        = row["GPPct12M"]
    customers     = row["Customers12M"]
    stock_age     = row["AvgStockAgeDays"]
    abc           = row["ABCClass"]
    strategic     = row["StrategicFlag"]
    lead_time     = row["LeadTimeDays"]

    if sales12m == 0 and available_qty > 0:   score += 35  # Dead stock
    if sales6m == 0 and sales12m > 0:         score += 15  # Declining demand
    if months_cover > 6:                       score += 20  # High overstock
    elif months_cover > 3:                     score += 10  # Moderate overstock
    if gp_pct < 0.10:                         score += 15  # Low margin
    elif gp_pct < 0.20:                       score += 8   # Medium margin
    if customers <= 1:                         score += 10  # Single customer risk
    elif customers <= 3:                       score += 5   # Limited spread
    if stock_age > 180:                        score += 15  # Aging stock
    elif stock_age > 90:                       score += 8   # Moderately aged
    if abc == "C":                             score += 10  # Low value product
    elif abc == "B":                           score += 4   # Medium value product
    if strategic in ["yes", "true", "1"]:     score -= 20  # Strategic protection
    if lead_time >= 45:                        score += 5   # Long replenishment

    return max(0, min(score, 100))

df["RiskScore"] = df.apply(calculate_risk, axis=1)

# ========================
# RISK LEVEL
# ========================
def classify_risk(score):
    if score >= 70:   return "High"
    elif score >= 40: return "Medium"
    return "Low"

df["RiskLevel"] = df["RiskScore"].apply(classify_risk)

# ========================
# AI RECOMMENDATION
# ========================
def recommendation(row):
    if row["RiskLevel"] == "High":
        if row["Sales12M"] == 0 and row["AvailableQty"] > 0:
            return "Discontinue or liquidate stock"
        if row["MonthsCover6M"] > 6:
            return "Reduce replenishment and clear excess stock"
        return "Immediate review required"
    if row["RiskLevel"] == "Medium":
        if row["GPPct12M"] < 0.10:
            return "Review pricing or margin"
        return "Monitor closely and optimize ordering"
    return "Stable product, maintain policy"

df["AI_Recommendation"] = df.apply(recommendation, axis=1)

# ========================
# CLUSTERING
# ========================
# 6 features selected for minimal overlap and maximum signal:
#   Sales12M       → absolute demand volume
#   MonthsCover6M  → stock vs demand balance
#   GPPct12M       → profitability
#   AvgStockAgeDays→ physical stock aging
#   Customers12M   → demand concentration risk
#   XYZ_Encoded    → demand predictability
#
# Excluded: Sales6M (correlated), AvailableQty (in MonthsCover),
#           LeadTimeDays (supplier not product), ABCClass (from Sales12M),
#           RiskScore (composite of above features)

cluster_features = [
    "Sales12M",
    "MonthsCover6M",
    "GPPct12M",
    "AvgStockAgeDays",
    "Customers12M",
    "XYZ_Encoded"
]

X = df[cluster_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["ClusterID"] = kmeans.fit_predict(X_scaled)

# ========================
# CLUSTER SUMMARY
# ========================
# Always review this before confirming labels below.
# ClusterID numbers are assigned arbitrarily — verify after every rerun.
cluster_summary = df.groupby("ClusterID")[cluster_features].mean().round(2)
print("\n=== Cluster Summary (verify before confirming labels) ===")
print(cluster_summary)

# ========================
# CLUSTER LABELS
# ========================
# Labels based on last verified run. Update if data or features change.
# Top Performers       → #0070C0 Blue
# Active / Overstocked → #00B050 Green
# Slow Movers          → #FFA500 Orange
# Dead Stock / Aging   → #FF0000 Red
# Extreme Overstock    → #8B0000 Dark Red

cluster_label_map = {
    0: "Active / Overstocked",
    1: "Dead Stock / Aging",
    2: "Top Performers",
    3: "Slow Movers / Excess Stock",
    4: "Extreme Overstock"
}

df["ClusterLabel"] = df["ClusterID"].map(cluster_label_map)

# ========================
# SAVE OUTPUT
# ========================
output = df[[
    "ProductID", "ProductName",
    "RiskScore", "RiskLevel", "AI_Recommendation",
    "ClusterID", "ClusterLabel"
]].copy()

output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"\nDone: {OUTPUT_PATH} created")
print("\nCluster distribution:")
print(df["ClusterLabel"].value_counts())
print("\nXYZ distribution across clusters:")
print(df.groupby(["ClusterLabel", "XYZ_Combined"]).size().unstack(fill_value=0))
