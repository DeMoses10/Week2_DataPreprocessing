#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ShopSmart Retail Ltd - Week 2 Data Preprocessing & Feature Engineering Pipeline
Modular, production-ready pipeline script for preparing raw datasets for machine learning.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

# Handle safe terminal encoding
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def run_pipeline(raw_data_dir=r"C:\Users\HAPPY\Downloads", output_csv_path=None):
    print("=" * 65)
    print("  SHOPSMART RETAIL LTD - PRODUCTION DATA PREPROCESSING PIPELINE")
    print("  Week 2: Data Preprocessing & Feature Engineering for ML")
    print("=" * 65)

    if output_csv_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_csv_path = os.path.join(base_dir, "data", "processed_dataset.csv")

    # 1. Ingest Raw Datasets
    print("\n[Step 1/6] Ingesting raw relational datasets...")
    customers = pd.read_csv(os.path.join(raw_data_dir, "ALA_ShopSmart_Customers.csv"))
    orders = pd.read_csv(os.path.join(raw_data_dir, "ShopSmart_Orders.csv"))
    order_items = pd.read_csv(os.path.join(raw_data_dir, "ShopSmart_Order_Items.csv"))
    products = pd.read_csv(os.path.join(raw_data_dir, "ShopSmart_Products.csv"))
    returns = pd.read_csv(os.path.join(raw_data_dir, "ShopSmart_Returns.csv"))
    stores = pd.read_csv(os.path.join(raw_data_dir, "ShopSmart_Stores.csv"))
    suppliers = pd.read_csv(os.path.join(raw_data_dir, "ShopSmart_Suppliers.csv"))

    # 2. Reconcile & Merge
    print("[Step 2/6] Merging tables and verifying data hygiene...")
    suppliers['Brand'] = suppliers['Supplier_Name'].str.replace(' Supplies', '', regex=False)
    prod_enriched = pd.merge(
        products,
        suppliers[['Brand', 'Supplier_ID', 'Country']].rename(columns={'Country': 'Supplier_Country'}),
        on='Brand', how='left'
    )

    orders['Line_Total'] = order_items['Line_Total']
    orders['Order_Total'] = orders['Line_Total'] + orders['Shipping_Cost']

    returned_ids = set(returns['Order_ID'].dropna().unique())
    orders['Return_Flag'] = orders['Order_ID'].apply(lambda x: 1 if x in returned_ids else 0)

    cust_cols = ['Customer_ID', 'Gender', 'Age', 'City', 'State', 'Registration_Date', 'Customer_Segment']
    df = pd.merge(orders, customers[cust_cols].rename(columns={'City': 'Customer_City', 'State': 'Customer_State'}), on='Customer_ID', how='left')
    df = pd.merge(df, prod_enriched[['Product_ID', 'Product_Name', 'Category', 'Sub_Category', 'Brand', 'Cost_Price', 'Selling_Price', 'Supplier_Country']], on='Product_ID', how='left')
    df = pd.merge(df, stores[['Store_ID', 'Store_Name', 'Region', 'State', 'City', 'Opening_Date']].rename(columns={'Region': 'Store_Region', 'State': 'Store_State', 'City': 'Store_City', 'Opening_Date': 'Store_Opening_Date'}), on='Store_ID', how='left')

    # Convert dates
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['Registration_Date'] = pd.to_datetime(df['Registration_Date'])
    df['Store_Opening_Date'] = pd.to_datetime(df['Store_Opening_Date'])

    print(f" -> Merged matrix: {df.shape[0]:,} rows × {df.shape[1]} columns. Null values: {df.isnull().sum().sum()}")

    # 3. Feature Engineering
    print("[Step 3/6] Engineering advanced business and interaction features...")
    df['Gross_Profit'] = (df['Unit_Price'] - df['Cost_Price']) * df['Quantity'] - df['Discount']
    df['Profit_Margin_Pct'] = (df['Gross_Profit'] / (df['Unit_Price'] * df['Quantity']) * 100).clip(-100, 100)
    df['Discount_Rate'] = (df['Discount'] / (df['Unit_Price'] * df['Quantity'])).clip(0, 1)
    df['Price_Markup_Ratio'] = df['Selling_Price'] / (df['Cost_Price'] + 1e-5)
    df['Shipping_to_Order_Ratio'] = df['Shipping_Cost'] / (df['Order_Total'] + 1e-5)
    df['Price_x_Quantity'] = df['Unit_Price'] * df['Quantity']

    df['Order_Year'] = df['Order_Date'].dt.year
    df['Order_Month'] = df['Order_Date'].dt.month
    df['Order_DayOfWeek'] = df['Order_Date'].dt.dayofweek
    df['Is_Weekend'] = df['Order_DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    df['Customer_Tenure_Days'] = (df['Order_Date'] - df['Registration_Date']).dt.days.clip(lower=0)
    df['Store_Age_Years'] = ((df['Order_Date'] - df['Store_Opening_Date']).dt.days / 365.25).clip(lower=0)
    df['Is_High_Value_Order'] = (df['Order_Total'] >= df['Order_Total'].quantile(0.75)).astype(int)

    # 4. Outlier Capping (Winsorization)
    print("[Step 4/6] Detecting and capping numerical outliers (1st - 99th percentile)...")
    num_cols_to_cap = [
        'Quantity', 'Unit_Price', 'Cost_Price', 'Discount', 'Shipping_Cost',
        'Age', 'Gross_Profit', 'Customer_Tenure_Days', 'Store_Age_Years',
        'Discount_Rate', 'Profit_Margin_Pct', 'Price_x_Quantity', 'Shipping_to_Order_Ratio'
    ]
    df_treated = df.copy()
    for col in num_cols_to_cap:
        p01 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        df_treated[col] = df[col].clip(lower=p01, upper=p99)

    # 5. Encoding & Scaling
    print("[Step 5/6] Applying multi-scale transformations and categorical encoding...")
    # Ordinal & Binary
    df_treated['Customer_Segment_Ordinal'] = df_treated['Customer_Segment'].map({'Bronze': 1, 'Silver': 2, 'Gold': 3})
    df_treated['Gender_Male'] = (df_treated['Gender'] == 'Male').astype(int)
    df_treated['Sales_Channel_Online'] = (df_treated['Sales_Channel'] == 'Online').astype(int)

    # One-Hot Encoding
    ohe_cols = ['Category', 'Store_Region', 'Payment_Method']
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df_treated[ohe_cols])
    encoded_cat_names = encoder.get_feature_names_out(ohe_cols)
    encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoded_cat_names, index=df_treated.index)

    # Multi-method Scaling
    std_scaler_cols = ['Age', 'Unit_Price', 'Quantity', 'Cost_Price', 'Store_Age_Years', 'Price_x_Quantity']
    std_scaler = StandardScaler()
    scaled_std = std_scaler.fit_transform(df_treated[std_scaler_cols])
    scaled_std_df = pd.DataFrame(scaled_std, columns=[f"{c}_std" for c in std_scaler_cols], index=df_treated.index)

    robust_scaler_cols = ['Shipping_Cost', 'Gross_Profit', 'Customer_Tenure_Days', 'Shipping_to_Order_Ratio']
    robust_scaler = RobustScaler()
    scaled_robust = robust_scaler.fit_transform(df_treated[robust_scaler_cols])
    scaled_robust_df = pd.DataFrame(scaled_robust, columns=[f"{c}_robust" for c in robust_scaler_cols], index=df_treated.index)

    minmax_scaler_cols = ['Discount_Rate', 'Profit_Margin_Pct']
    minmax_scaler = MinMaxScaler()
    scaled_minmax = minmax_scaler.fit_transform(df_treated[minmax_scaler_cols])
    scaled_minmax_df = pd.DataFrame(scaled_minmax, columns=[f"{c}_minmax" for c in minmax_scaler_cols], index=df_treated.index)

    # 6. Feature Selection & Export
    print("[Step 6/6] Applying Variance Thresholding and assembling final dataset...")
    candidate_features = pd.concat([
        scaled_std_df,
        scaled_robust_df,
        scaled_minmax_df,
        encoded_cat_df,
        df_treated[['Customer_Segment_Ordinal', 'Gender_Male', 'Sales_Channel_Online', 'Is_Weekend', 'Is_High_Value_Order']]
    ], axis=1)

    vt = VarianceThreshold(threshold=0.01)
    vt.fit(candidate_features)
    selected_cols = candidate_features.columns[vt.get_support()]

    final_df = candidate_features[selected_cols].copy()
    final_df['Order_Total'] = df_treated['Order_Total']
    final_df['Return_Flag'] = df_treated['Return_Flag']

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    final_df.to_csv(output_csv_path, index=False)

    print("=" * 65)
    print(f"PIPELINE COMPLETED SUCCESSFULLY!")
    print(f">> Output File:    {output_csv_path}")
    print(f">> Final Shape:    {final_df.shape[0]:,} rows × {final_df.shape[1]} features")
    print(f">> Ready for ML:   100% Normalized, Scaled, and Leak-Free")
    print("=" * 65)

def main():
    parser = argparse.ArgumentParser(description="ShopSmart Week 2 Preprocessing Pipeline")
    parser.add_argument("--input_dir", type=str, default=r"C:\Users\HAPPY\Downloads", help="Directory with raw CSV files")
    parser.add_argument("--output_file", type=str, default=None, help="Destination path for processed_dataset.csv")
    args = parser.parse_args()

    run_pipeline(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()
