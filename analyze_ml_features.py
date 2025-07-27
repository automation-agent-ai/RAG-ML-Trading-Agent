#!/usr/bin/env python3
"""
Analyze ML Feature Tables
"""

import pandas as pd
import os
from pathlib import Path
import glob

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
    
    # Check for non-NaN values in key columns
    if 'total_revenue' in df.columns:
        non_nan_revenue = df['total_revenue'].notna().sum()
        print(f"Rows with non-NaN total_revenue: {non_nan_revenue} ({non_nan_revenue/len(df):.1%})")
    
    if 'count_financial_results' in df.columns:
        non_nan_news = df['count_financial_results'].notna().sum()
        print(f"Rows with non-NaN count_financial_results: {non_nan_news} ({non_nan_news/len(df):.1%})")
    
    # Check for label column
    if 'label' in df.columns:
        non_nan_label = df['label'].notna().sum()
        print(f"Rows with non-NaN label: {non_nan_label} ({non_nan_label/len(df):.1%})")
        
        if non_nan_label > 0:
            print(f"Label statistics:")
            print(df['label'].describe())
    
    # Sample rows with non-NaN values
    print("\nSample row with non-NaN values:")
    
    # For financial features
    if 'total_revenue' in df.columns:
        non_nan_df = df[df['total_revenue'].notna()]
        if len(non_nan_df) > 0:
            sample_cols = ['setup_id', 'total_revenue', 'gross_profit', 'net_income', 
                          'debt_to_equity', 'roe', 'revenue_growth_yoy']
            sample_cols = [col for col in sample_cols if col in df.columns]
            print(non_nan_df[sample_cols].head(1).to_string())
    
    # For text features
    if 'count_financial_results' in df.columns:
        non_nan_df = df[df['count_financial_results'].notna()]
        if len(non_nan_df) > 0:
            sample_cols = ['setup_id', 'count_financial_results', 'sentiment_score_financial_results',
                          'posts_avg_sentiment', 'post_count', 'recommendation_count']
            sample_cols = [col for col in sample_cols if col in df.columns]
            print(non_nan_df[sample_cols].head(1).to_string())

def main():
    """Main function"""
    # Find all CSV files in the ml_features directory
    csv_files = glob.glob('data/ml_features/*.csv')
    
    if not csv_files:
        print("No CSV files found in data/ml_features/")
        return
    
    # Analyze each CSV file
    for file_path in csv_files:
        analyze_csv(file_path)

if __name__ == '__main__':
    main() 