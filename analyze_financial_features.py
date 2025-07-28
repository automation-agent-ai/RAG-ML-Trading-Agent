#!/usr/bin/env python3
"""
Analyze Financial Features CSV
"""

import pandas as pd
import glob
import os
from pathlib import Path

def analyze_csv(file_path):
    """Analyze a CSV file and print summary statistics"""
    print(f"\n{'='*50}")
    print(f"Analyzing: {os.path.basename(file_path)}")
    print(f"{'='*50}")
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Basic info
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Rows: {len(df)}")
    
    # Check for non-null values in key columns
    print("\nColumns with non-null values:")
    for col in df.columns:
        non_null = df[col].notna().sum()
        if non_null > 0 and col != 'setup_id':
            print(f"- {col}: {non_null} ({non_null/len(df):.1%})")
    
    # Check for label column
    if 'label' in df.columns:
        non_nan_label = df['label'].notna().sum()
        print(f"\nRows with non-NaN label: {non_nan_label} ({non_nan_label/len(df):.1%})")
        
        if non_nan_label > 0:
            print(f"Label statistics:")
            print(df['label'].describe())
    
    # Sample rows with non-NaN values
    print("\nSample row with non-NaN financial values:")
    
    # For financial features
    if 'total_revenue' in df.columns:
        non_nan_df = df[df['total_revenue'].notna()]
        if len(non_nan_df) > 0:
            sample_cols = ['setup_id', 'total_revenue', 'gross_profit', 'net_income', 
                          'debt_to_equity', 'roe', 'revenue_growth_yoy']
            sample_cols = [col for col in sample_cols if col in df.columns]
            print(non_nan_df[sample_cols].head(1).to_string())
    
    # Export a clean version with only rows that have financial data
    if 'total_revenue' in df.columns:
        non_nan_df = df[df['total_revenue'].notna()]
        if len(non_nan_df) > 0:
            clean_file = file_path.replace('.csv', '_clean.csv')
            non_nan_df.to_csv(clean_file, index=False)
            print(f"\nExported clean version with {len(non_nan_df)} rows to {os.path.basename(clean_file)}")

def main():
    """Main function"""
    # Find all financial ML features CSV files
    csv_files = glob.glob('data/ml_features/financial_ml_features_*.csv')
    
    if not csv_files:
        print("No financial ML features CSV files found")
        return
    
    # Sort by modification time (newest first)
    csv_files.sort(key=os.path.getmtime, reverse=True)
    
    # Analyze the most recent file
    analyze_csv(csv_files[0])

if __name__ == '__main__':
    main() 