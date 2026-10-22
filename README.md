# 🔬 ShopSmart Retail Ltd. — Data Preprocessing & Feature Engineering

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![AnalystLab Africa](https://img.shields.io/badge/AnalystLab%20Africa-Week%202%20ML-1E3A8A?style=flat)](https://analystlab.africa)
[![Status](https://img.shields.io/badge/Status-Completed-059669?style=flat)]()

**Official Weekly Assignment — Week 2**  
**Programme:** Machine Learning Internship Programme  
**Client:** ShopSmart Retail Ltd.  
**Author:** Junior Machine Learning Engineer

---

## 📌 Business Scenario & Objective

Real-world datasets often contain missing values, inconsistent formats, categorical variables, outliers, and irrelevant features that degrade machine learning performance.

Following our exploratory analysis in Week 1, our mandate for **ShopSmart Retail Ltd.** is to construct an end-to-end, reproducible data preprocessing and feature engineering pipeline on 50,000 historical sales transactions, transforming raw departmental tables into a high-signal, machine-learning-ready matrix.

---

## 📂 Repository Structure & Deliverables

```text
Week2_DataPreprocessing/
├── ML02_Data_Preprocessing.ipynb        # Complete, executed end-to-end Jupyter Notebook with all outputs
├── data/
│   ├── processed_dataset.csv            # Final ML-ready dataset (50,000 rows × 28 features)
│   └── outlier_analysis.csv             # Statistical breakdown of IQR and Z-score outlier audits
├── reports/
│   ├── Business_Understanding_Report.pdf # Formal business context, stakeholder ROI, and objectives
│   ├── Data_Preprocessing_Report.pdf    # Technical report detailing every preprocessing decision
│   └── assets/                          # 8 High-resolution diagnostic figures and charts
├── scripts/
│   └── preprocess_pipeline.py           # Production CLI pipeline script for automated preprocessing
├── requirements.txt                     # Pinned project dependencies
├── README.md                            # Comprehensive technical documentation
└── .gitignore                           # Standard git exclusion rules
```

---

## 💡 Answers to the 7 Core Business Questions

| # | Business Question | Comprehensive Technical Finding |
| :-: | :--- | :--- |
| **1** | **What data quality issues exist?** | Raw tables were structurally complete (zero nulls, zero duplicates), but featured: (a) unparsed string dates; (b) high-cardinality PII identifiers; (c) positive skewness in financial values (`Gross_Profit`, `Shipping_Cost`); and (d) acute class imbalance (4.88% returns). |
| **2** | **Which features require preprocessing?** | Categoricals (`Customer_Segment`, `Gender`, `Sales_Channel`, `Category`, `Store_Region`, `Payment_Method`) required encoding; date fields required conversion into tenure/age signals; and continuous features required outlier capping and scaling. |
| **3** | **Which encoding techniques are most appropriate?** | **Ordinal Encoding** for `Customer_Segment` (preserving Bronze &lt; Silver &lt; Gold); **Binary Mapping** for `Gender` and `Sales_Channel`; and **One-Hot Encoding (`drop='first'`)** for nominal multi-class variables to avoid multicollinearity. |
| **4** | **Which scaling method should be applied?** | **StandardScaler** for Gaussian features (`Age`, `Unit_Price`, `Quantity`, `Cost_Price`); **RobustScaler** for heavy-tailed financial metrics (`Gross_Profit`, `Shipping_Cost`, `Customer_Tenure_Days`); and **MinMaxScaler** for bounded percentage ratios (`Discount_Rate`, `Profit_Margin_Pct`). |
| **5** | **Are there significant outliers that need treatment?** | Yes, in `Gross_Profit` and `Shipping_Cost`. Rather than dropping records, we applied **1st and 99th percentile Winsorization (capping)**, eliminating extreme leverage while preserving all 50,000 orders. |
| **6** | **Which features contribute most to the business problem?** | For Revenue Forecasting (`Order_Total`), the engineered interaction `Price_x_Quantity` ($|r| = 0.999$), `Gross_Profit` ($|r| = 0.868$), and `Is_High_Value_Order` ($|r| = 0.843$) dominate. For Returns (`Return_Flag`), `Store_Region_North`, `Shipping_Cost`, and `Category_Groceries` provide the strongest signals. |
| **7** | **Is the dataset ready for model training?** | **Yes.** `data/processed_dataset.csv` contains 50,000 clean records across 28 normalized, non-redundant numerical features with zero missing values, ready for estimator training. |

---

## ⚙️ Advanced Feature Engineering (13 New Features)

1. **`Order_Total`**: $Line\_Total + Shipping\_Cost$ (Grand total billing target).
2. **`Gross_Profit`**: $(Unit\_Price - Cost\_Price) \times Quantity - Discount$ (Basket profit contribution).
3. **`Profit_Margin_Pct`**: Ratio of gross profit to gross sales, clipped to $[-100\%, +100\%]$.
4. **`Discount_Rate`**: Proportion of unit price discounted, bounded to $[0, 1]$.
5. **`Price_Markup_Ratio`**: Multiplier ratio of selling price over cost price.
6. **`Price_x_Quantity`**: Interaction subtotal before discounts.
7. **`Shipping_to_Order_Ratio`**: Shipping friction index ($Shipping\_Cost / Order\_Total$).
8. **`Customer_Tenure_Days`**: Elapsed days between customer registration and purchase date.
9. **`Store_Age_Years`**: Operational maturity of store location at transaction date.
10. **`Order_Month`**: Seasonal monthly indicator.
11. **`Order_DayOfWeek`**: Day of the week (0 to 6).
12. **`Is_Weekend`**: Binary flag isolating weekend purchasing behavior.
13. **`Is_High_Value_Order`**: Binary indicator for orders in the top 25th percentile of spend.

---

## 📊 Visualizations Included

All charts are saved at high resolution (300 DPI) in `reports/assets/`:
1. **Target Distributions (`1_target_distributions.png`)**: Histograms of continuous `Order_Total` and binary `Return_Flag`.
2. **Outlier Box Plots (`2_outlier_boxplots.png`)**: Before vs. After Winsorization on `Gross_Profit` and `Shipping_Cost`.
3. **Correlation Heatmap (`3_correlation_heatmap.png`)**: Full Pearson correlation matrix across engineered features.
4. **Categorical Distributions (`4_categorical_distributions.png`)**: Count plots by Category, Loyalty Tier, Channel, and Payment.
5. **Scatter Relationships (`5_scatter_relationships.png`)**: Interaction term and gross profit vs order spend.
6. **Feature Selection Ranking (`6_feature_selection_ranking.png`)**: Top 10 correlated features for model development.
7. **Customer Tenure vs. Spend (`7_tenure_vs_spend.png`)**: Spend distribution across tenure quintiles.
8. **Monthly Sales Trend (`8_monthly_sales_trend.png`)**: Seasonal volume trajectories by product department.

---

## 🚀 Running the Automated Preprocessing Pipeline

You can run the standalone preprocessing pipeline via command line:

```bash
# Run pipeline with default inputs and outputs
python scripts/preprocess_pipeline.py

# Or specify custom directories
python scripts/preprocess_pipeline.py --input_dir "path/to/raw/csvs" --output_file "data/processed_dataset.csv"
```

---

## 🛠️ Environment Setup & Reproduction

```bash
# 1. Clone the repository
git clone <YOUR_REPO_URL>
cd Week2_DataPreprocessing

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Jupyter Notebook
jupyter notebook ML02_Data_Preprocessing.ipynb
```

---

## 📄 License & Attribution
Developed as part of the **AnalystLab Africa Machine Learning Internship Programme (Week 2)** for **ShopSmart Retail Ltd.** All rights reserved.
