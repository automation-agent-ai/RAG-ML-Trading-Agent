#!/usr/bin/env python3
"""
Merge Ensemble Predictions with ML Features

This script merges the ensemble predictions with the text ML features table,
creating a comprehensive dataset for machine learning.

Usage:
    python merge_ensemble_with_features.py --ensemble data/ensemble_predictions.csv --text-features data/ml_features/text_ml_features_prediction_20250728_015210.csv --output data/merged_ml_features.csv
"""

import os
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def merge_ensemble_with_features(
    ensemble_file: str,
    text_features_file: str,
    financial_features_file: str = None,
    output_file: str = "data/merged_ml_features.csv"
):
    """
    Merge ensemble predictions with text ML features and optionally financial features
    
    Args:
        ensemble_file: Path to ensemble predictions CSV
        text_features_file: Path to text ML features CSV
        financial_features_file: Path to financial ML features CSV (optional)
        output_file: Path to output merged CSV
    """
    logger.info(f"Loading ensemble predictions from {ensemble_file}")
    ensemble_df = pd.read_csv(ensemble_file)
    
    logger.info(f"Loading text ML features from {text_features_file}")
    text_df = pd.read_csv(text_features_file)
    
    # Merge ensemble predictions with text features
    logger.info("Merging ensemble predictions with text features")
    merged_df = pd.merge(text_df, ensemble_df, on="setup_id", how="left")
    
    # Optionally merge with financial features
    if financial_features_file:
        logger.info(f"Loading financial ML features from {financial_features_file}")
        financial_df = pd.read_csv(financial_features_file)
        
        logger.info("Merging with financial features")
        # Drop setup_id from financial_df to avoid duplicate columns
        financial_df_no_id = financial_df.drop(columns=["setup_id"])
        
        # Merge with financial features
        merged_df = pd.merge(
            merged_df,
            financial_df,
            on="setup_id",
            how="left",
            suffixes=("", "_financial")
        )
    
    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    merged_df.to_csv(output_file, index=False)
    logger.info(f"Saved merged ML features to {output_file}")
    
    # Print summary
    logger.info("\nMerged Dataset Summary:")
    logger.info(f"- Rows: {len(merged_df)}")
    logger.info(f"- Columns: {len(merged_df.columns)}")
    
    # Count non-null values in key columns
    ensemble_cols = ["predicted_outperformance", "predicted_class", "confidence"]
    for col in ensemble_cols:
        if col in merged_df.columns:
            non_null = merged_df[col].notna().sum()
            logger.info(f"- Rows with non-null {col}: {non_null} ({non_null/len(merged_df):.1%})")
    
    return merged_df

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Merge ensemble predictions with ML features')
    parser.add_argument('--ensemble', required=True,
                       help='Path to ensemble predictions CSV')
    parser.add_argument('--text-features', required=True,
                       help='Path to text ML features CSV')
    parser.add_argument('--financial-features',
                       help='Path to financial ML features CSV')
    parser.add_argument('--output', default='data/merged_ml_features.csv',
                       help='Path to output merged CSV')
    
    args = parser.parse_args()
    
    merge_ensemble_with_features(
        ensemble_file=args.ensemble,
        text_features_file=args.text_features,
        financial_features_file=args.financial_features,
        output_file=args.output
    )

if __name__ == '__main__':
    main() 