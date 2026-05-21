# AI-Powered Inventory & Sales Decision Engine

A Power BI dashboard that combines machine learning and business rules
to help inventory and supply chain teams move faster from data to decision.

---

## The Problem

Most inventory dashboards tell you what happened.
This one tells you what to do about it.

Managing hundreds of products means constant judgment calls — what to
restock, what to liquidate, what to watch. This dashboard automates that
thinking using AI, so teams can focus on action instead of analysis.

---

## What It Does

Every product in the portfolio is automatically assessed across five dimensions:

- **Risk Score** — how likely is this product to become a problem?
- **Cluster** — what behavioral group does this product belong to?
- **Forecast** — what does demand look like over the next 6 months?
- **ABC-XYZ Class** — how valuable is it, and how predictable is its demand?
- **Decision** — what should you do with it right now?

---

## Product Decisions

Every product receives one of five decisions, calculated automatically:

| Decision | Meaning |
|---|---|
| **Active** | Healthy product — maintain current policy |
| **Watch** | Needs monitoring — review ordering strategy |
| **Watch / Unpredictable** | Demand is too erratic to plan reliably — handle with caution |
| **Excess** | Overstocked relative to demand — reduce replenishment |
| **Obsolete** | No recent sales with stock on hand — consider liquidation |

---

## Dashboard Pages

| Page | What It Shows |
|---|---|
| 01 Executive Overview | High-level KPIs, sales trend, revenue by category |
| 02 Product Decision Engine | Product-level decisions and AI recommendations |
| 03 Inventory Intelligence | Stock health, demand vs supply, aging analysis |
| 04 Smart Recommendations | Prioritized action list by decision type |
| 05 Key Drivers Analysis | What factors influence each product decision |
| 06 Cluster & Risk Intelligence | AI clustering results and risk distribution |
| 07 Product Forecast | 6-month demand forecast per product |
| 08 Today's Actions | Daily triage page |
| 09 ABC-XYZ Analysis | Portfolio segmentation matrix |

---

## Screenshots

### 01 — Executive Overview
![Executive Overview](screenshots/01_executive_overview.png)

### 04 — Smart Recommendations
![Smart Recommendations](screenshots/04_smart_recommendations.png)

### 06 — Cluster & Risk Intelligence
![Cluster & Risk Intelligence](screenshots/06_Cluster___Risk_Intelligence.png)

### 07 — Product Forecast
![Product Forecast](screenshots/07_Product_Forecast.png)

---

## Python Scripts

Three scripts run outside Power BI and feed results back into the dashboard.

### `risk_scoring_clustering.py`
Calculates a risk score (0–100) for every product based on sales patterns,
stock age, margin, and customer concentration. Then runs KMeans clustering
to group products into 5 behavioral segments:

| Cluster | Description |
|---|---|
| Top Performers | High sales, healthy stock, predictable demand |
| Active / Overstocked | Good sales but too much inventory |
| Slow Movers / Excess Stock | Low sales, high stock cover |
| Dead Stock / Aging | No recent sales, physically aging inventory |
| Extreme Overstock | Critical overstock — months of cover exceeds 1,000 |

### `forecast_prophet.py`
Uses Meta's Prophet model to forecast demand for the next 6 months per product.
Outputs forecast values with 95% confidence intervals (yhat_lower, yhat_upper).
A wide confidence interval indicates unpredictable demand — consistent with Z classification.

### `xyz_classification.py`
Calculates the Coefficient of Variation (CV) for each product across revenue and quantity.
Classifies demand predictability as X, Y, or Z.
Uses the worse of the two CV scores for the final classification.

| Class | CV Range | Meaning |
|---|---|---|
| X | < 0.5 | Stable, predictable — plan with confidence |
| Y | 0.5 – 1.0 | Moderate variability — plan with seasonal awareness |
| Z | > 1.0 | Erratic, unpredictable — handle with caution |

---

## How It Works

```
Power BI (export visual)
        ↓
CSV files
        ↓
Python scripts (risk scoring + clustering + forecasting + XYZ)
        ↓
Output CSV files
        ↓
Power BI (import + refresh)
        ↓
Dashboard pages
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Power BI | Dashboard, DAX measures, data model |
| Python | Machine learning and forecasting |
| scikit-learn | KMeans clustering, StandardScaler |
| Prophet | Demand forecasting |
| pandas | Data processing |
| DAX | Business logic and decision rules |

---

## Repo Structure

```
ai-inventory-decision-engine/
│
├── data/
│   ├── product_ai_input.csv          ← export from Power BI (clustering input)
│   ├── sales_timeseries.csv          ← export from Power BI (forecast input)
│   └── ai_powered_inventory_sales_engine_dataset.xlsx  ← source data
│
├── screenshots/
│   ├── 01_executive_overview.png
│   ├── 04_smart_recommendations.png
│   ├── 06_Cluster___Risk_Intelligence.png
│   └── 07_Product_Forecast.png
│
├── risk_scoring_clustering.py
├── forecast_prophet.py
├── xyz_classification.py
└── README.md
```

---

## How to Run

1. Clone the repo
2. Install dependencies:
```bash
pip install pandas scikit-learn prophet openpyxl
```
3. Place your input files in the `data/` folder
4. Run the scripts in this order:
```bash
python xyz_classification.py
python risk_scoring_clustering.py
python forecast_prophet.py
```
5. Load the output CSVs back into Power BI and refresh

---

## Author

Built by Lana Al Maradni
Data Analyst | AI-Augmented Analytics
Dubai, UAE
[LinkedIn](https://www.linkedin.com/in/lanamaradni)

---

*The data has more stories to tell — and I'm just getting started.*
