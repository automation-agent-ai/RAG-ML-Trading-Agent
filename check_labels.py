#!/usr/bin/env python3
"""
Check Labels in CSV Files

This script checks the label columns in CSV files.
"""

import os
import sys
import pandas as pd
import glob
from pathlib import Path

def check_labels(file_path):
    """Check labels in a CSV file"""
    print(f"\nChecking {os.path.basename(file_path)}")
    print("-" * 50)
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Basic info
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    # Check for label column
    if 'label' in df.columns:
        non_null = df['label'].notna().sum()
        print(f"Rows with non-null label: {non_null} ({non_null/len(df):.1%})")
        
        if non_null > 0:
            # Get unique values
            unique_values = df['label'].dropna().unique()
            print(f"Unique label values: {unique_values}")
            
            # Get value counts
            value_counts = df['label'].value_counts(dropna=False)
            print("\nLabel value counts:")
            print(value_counts)
    else:
        print("No 'label' column found")
    
    # Check for other potential label columns
    potential_label_cols = [col for col in df.columns if 'label' in col.lower() or 'class' in col.lower() or 'target' in col.lower()]
    if potential_label_cols:
        print("\nOther potential label columns:")
        for col in potential_label_cols:
            non_null = df[col].notna().sum()
            print(f"- {col}: {non_null} non-null values ({non_null/len(df):.1%})")
    
    # Check for outperformance column
    if 'outperformance_10d' in df.columns:
        non_null = df['outperformance_10d'].notna().sum()
        print(f"\nRows with non-null outperformance_10d: {non_null} ({non_null/len(df):.1%})")
        
        if non_null > 0:
            # Get statistics
            print("\noutperformance_10d statistics:")
            print(df['outperformance_10d'].describe())

def main():
    """Main function"""
    # Check all CSV files in the ml_features directory
    csv_files = glob.glob('data/ml_features/*.csv')
    
    if not csv_files:
        print("No CSV files found in data/ml_features")
        return
    
    # Sort by modification time (newest first)
    csv_files.sort(key=os.path.getmtime, reverse=True)
    
    # Check all files
    for file_path in csv_files:
        check_labels(file_path)

if __name__ == '__main__':
    main() 