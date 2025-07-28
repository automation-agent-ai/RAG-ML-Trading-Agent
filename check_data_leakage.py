#!/usr/bin/env python3
"""
Check for Data Leakage in ML Pipeline

This script analyzes ML features and labels to identify potential data leakage issues.

Usage:
    python check_data_leakage.py --text-data data/ml_features/text_ml_features_training_labeled.csv 
                                --financial-data data/ml_features/financial_ml_features_training_labeled.csv
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLeakageChecker:
    """Class for checking data leakage in ML pipeline"""
    
    def __init__(self, output_dir: str = "data/leakage_analysis"):
        """
        Initialize the checker
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with data
        """
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Basic info
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        return df
    
    def check_target_correlation(self, df: pd.DataFrame, label_col: str = "label", 
                                output_file: str = None) -> pd.DataFrame:
        """
        Check correlation between features and target variable
        
        Args:
            df: DataFrame with features and labels
            label_col: Name of the label column
            output_file: Path to output file for correlation plot
            
        Returns:
            DataFrame with feature correlations
        """
        logger.info(f"Checking correlation with target variable '{label_col}'")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Remove label and target-related columns
        exclude_cols = [label_col, 'outperformance_10d', 'setup_id']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Calculate correlation with target
        correlations = []
        for col in feature_cols:
            # Skip columns with all NaN values
            if df[col].isna().all():
                continue
                
            corr = df[col].corr(df[label_col])
            correlations.append({
                'feature': col,
                'correlation': corr
            })
        
        # Create DataFrame and sort by absolute correlation
        corr_df = pd.DataFrame(correlations)
        corr_df['abs_correlation'] = corr_df['correlation'].abs()
        corr_df = corr_df.sort_values('abs_correlation', ascending=False)
        
        # Log top correlations
        logger.info("Top 10 feature correlations with target:")
        for _, row in corr_df.head(10).iterrows():
            logger.info(f"  - {row['feature']}: {row['correlation']:.4f}")
        
        # Plot correlation
        if output_file:
            self._plot_target_correlation(corr_df.head(20), output_file)
        
        return corr_df
    
    def _plot_target_correlation(self, corr_df: pd.DataFrame, output_file: str):
        """Plot correlation with target"""
        plt.figure(figsize=(12, 8))
        sns.barplot(x='correlation', y='feature', data=corr_df)
        plt.title('Feature Correlation with Target')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / output_file
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved correlation plot to {output_path}")
    
    def check_outperformance_correlation(self, df: pd.DataFrame, 
                                        output_file: str = None) -> pd.DataFrame:
        """
        Check correlation between features and outperformance
        
        Args:
            df: DataFrame with features and outperformance
            output_file: Path to output file for correlation plot
            
        Returns:
            DataFrame with feature correlations
        """
        if 'outperformance_10d' not in df.columns:
            logger.warning("No 'outperformance_10d' column found in data")
            return pd.DataFrame()
        
        logger.info("Checking correlation with outperformance")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Remove label and target-related columns
        exclude_cols = ['label', 'outperformance_10d', 'setup_id']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Calculate correlation with outperformance
        correlations = []
        for col in feature_cols:
            # Skip columns with all NaN values
            if df[col].isna().all():
                continue
                
            corr = df[col].corr(df['outperformance_10d'])
            correlations.append({
                'feature': col,
                'correlation': corr
            })
        
        # Create DataFrame and sort by absolute correlation
        corr_df = pd.DataFrame(correlations)
        corr_df['abs_correlation'] = corr_df['correlation'].abs()
        corr_df = corr_df.sort_values('abs_correlation', ascending=False)
        
        # Log top correlations
        logger.info("Top 10 feature correlations with outperformance:")
        for _, row in corr_df.head(10).iterrows():
            logger.info(f"  - {row['feature']}: {row['correlation']:.4f}")
        
        # Plot correlation
        if output_file:
            self._plot_outperformance_correlation(corr_df.head(20), output_file)
        
        return corr_df
    
    def _plot_outperformance_correlation(self, corr_df: pd.DataFrame, output_file: str):
        """Plot correlation with outperformance"""
        plt.figure(figsize=(12, 8))
        sns.barplot(x='correlation', y='feature', data=corr_df)
        plt.title('Feature Correlation with Outperformance')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / output_file
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved correlation plot to {output_path}")
    
    def check_feature_importance_vs_correlation(self, feature_importance_file: str, 
                                               corr_df: pd.DataFrame,
                                               output_file: str = None) -> pd.DataFrame:
        """
        Compare feature importance with target correlation
        
        Args:
            feature_importance_file: Path to feature importance CSV
            corr_df: DataFrame with feature correlations
            output_file: Path to output file for comparison plot
            
        Returns:
            DataFrame with comparison
        """
        logger.info(f"Comparing feature importance with target correlation")
        
        # Load feature importance
        try:
            importance_df = pd.read_csv(feature_importance_file)
            
            # Merge with correlation
            comparison_df = pd.merge(
                importance_df, 
                corr_df, 
                left_on='feature', 
                right_on='feature', 
                how='inner'
            )
            
            # Log comparison
            logger.info("Top 10 features by importance and correlation:")
            for _, row in comparison_df.head(10).iterrows():
                logger.info(f"  - {row['feature']}: importance={row['importance']:.4f}, correlation={row['correlation']:.4f}")
            
            # Plot comparison
            if output_file:
                self._plot_importance_vs_correlation(comparison_df, output_file)
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error loading feature importance: {e}")
            return pd.DataFrame()
    
    def _plot_importance_vs_correlation(self, comparison_df: pd.DataFrame, output_file: str):
        """Plot feature importance vs correlation"""
        plt.figure(figsize=(12, 8))
        plt.scatter(comparison_df['correlation'], comparison_df['importance'], alpha=0.6)
        
        # Add feature names for top points
        top_points = comparison_df.sort_values('importance', ascending=False).head(10)
        for _, row in top_points.iterrows():
            plt.annotate(
                row['feature'], 
                (row['correlation'], row['importance']),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.title('Feature Importance vs Target Correlation')
        plt.xlabel('Correlation with Target')
        plt.ylabel('Feature Importance')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / output_file
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved comparison plot to {output_path}")
    
    def check_setup_id_patterns(self, df: pd.DataFrame, label_col: str = "label") -> None:
        """
        Check for patterns in setup_id that might leak information
        
        Args:
            df: DataFrame with setup_id and labels
            label_col: Name of the label column
        """
        if 'setup_id' not in df.columns:
            logger.warning("No 'setup_id' column found in data")
            return
        
        logger.info("Checking for patterns in setup_id")
        
        # Extract components from setup_id (assuming format like "BBB_2024-04-29")
        df['ticker'] = df['setup_id'].str.split('_', expand=True)[0]
        
        # Try to extract date if available in the expected format
        try:
            df['date'] = df['setup_id'].str.split('_', expand=True)[1]
            
            # Check for date patterns - only for rows with valid dates
            valid_date_mask = df['date'].str.match(r'\d{4}-\d{2}-\d{2}')
            if valid_date_mask.any():
                date_df = df[valid_date_mask].copy()
                date_df['year'] = pd.to_datetime(date_df['date']).dt.year
                date_df['month'] = pd.to_datetime(date_df['date']).dt.month
                
                # Check correlation between month and label
                month_label_corr = date_df['month'].corr(date_df[label_col])
                logger.info(f"Correlation between month and label: {month_label_corr:.4f}")
                
                # Check correlation between year and label
                year_label_corr = date_df['year'].corr(date_df[label_col])
                logger.info(f"Correlation between year and label: {year_label_corr:.4f}")
            else:
                logger.warning("No valid dates found in setup_id")
        except Exception as e:
            logger.warning(f"Could not extract dates from setup_id: {e}")
        
        # Check correlation between ticker and label
        ticker_counts = df.groupby(['ticker', label_col]).size().unstack().fillna(0)
        ticker_distribution = ticker_counts.div(ticker_counts.sum(axis=1), axis=0)
        
        # Check if any ticker has strong bias towards a specific label
        bias_threshold = 0.8  # 80% of samples for a ticker have the same label
        biased_tickers = []
        
        for ticker, row in ticker_distribution.iterrows():
            if row.max() > bias_threshold and ticker_counts.loc[ticker].sum() >= 5:
                biased_tickers.append({
                    'ticker': ticker,
                    'dominant_label': row.idxmax(),
                    'percentage': row.max(),
                    'count': ticker_counts.loc[ticker].sum()
                })
        
        if biased_tickers:
            logger.warning(f"Found {len(biased_tickers)} tickers with strong label bias:")
            for item in biased_tickers:
                logger.warning(f"  - {item['ticker']}: {item['percentage']*100:.1f}% are label {item['dominant_label']} (n={item['count']})")
        else:
            logger.info("No strong ticker-label bias detected")
        
        # No additional date checks needed here - already handled in the try/except block above
    
    def check_train_test_overlap(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """
        Check for overlap between training and test data
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
        """
        logger.info("Checking for overlap between training and test data")
        
        if 'setup_id' not in train_df.columns or 'setup_id' not in test_df.columns:
            logger.warning("No 'setup_id' column found in data")
            return
        
        # Check for direct overlap
        train_ids = set(train_df['setup_id'])
        test_ids = set(test_df['setup_id'])
        overlap = train_ids.intersection(test_ids)
        
        if overlap:
            logger.warning(f"Found {len(overlap)} overlapping setup_ids between training and test data")
            logger.warning(f"Examples: {list(overlap)[:5]}")
        else:
            logger.info("No direct overlap between training and test data")
        
        # Check for ticker overlap
        if 'ticker' not in train_df.columns:
            train_df['ticker'] = train_df['setup_id'].str.split('_', expand=True)[0]
        
        if 'ticker' not in test_df.columns:
            test_df['ticker'] = test_df['setup_id'].str.split('_', expand=True)[0]
        
        train_tickers = set(train_df['ticker'])
        test_tickers = set(test_df['ticker'])
        ticker_overlap = train_tickers.intersection(test_tickers)
        
        logger.info(f"Found {len(ticker_overlap)} tickers present in both training and test data")
        logger.info(f"Ticker overlap: {len(ticker_overlap) / len(test_tickers):.1%} of test tickers")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Check for data leakage in ML pipeline')
    parser.add_argument('--text-data', required=True,
                      help='Path to text ML features CSV')
    parser.add_argument('--financial-data', required=True,
                      help='Path to financial ML features CSV')
    parser.add_argument('--text-test-data',
                      help='Path to text ML features test CSV')
    parser.add_argument('--financial-test-data',
                      help='Path to financial ML features test CSV')
    parser.add_argument('--text-importance',
                      help='Path to text feature importance CSV')
    parser.add_argument('--financial-importance',
                      help='Path to financial feature importance CSV')
    parser.add_argument('--output-dir', default='data/leakage_analysis',
                      help='Directory to save analysis results')
    parser.add_argument('--label-col', default='label',
                      help='Name of the label column')
    
    args = parser.parse_args()
    
    # Initialize checker
    checker = DataLeakageChecker(output_dir=args.output_dir)
    
    # Check text data
    logger.info("=== Checking Text Features ===")
    text_df = checker.load_data(args.text_data)
    
    # Check for target correlation
    text_corr = checker.check_target_correlation(
        text_df, 
        label_col=args.label_col,
        output_file='text_target_correlation.png'
    )
    
    # Check for outperformance correlation
    text_outperf_corr = checker.check_outperformance_correlation(
        text_df,
        output_file='text_outperformance_correlation.png'
    )
    
    # Check for setup_id patterns
    checker.check_setup_id_patterns(text_df, label_col=args.label_col)
    
    # Check feature importance vs correlation
    if args.text_importance:
        checker.check_feature_importance_vs_correlation(
            args.text_importance,
            text_corr,
            output_file='text_importance_vs_correlation.png'
        )
    
    # Check financial data
    logger.info("\n=== Checking Financial Features ===")
    financial_df = checker.load_data(args.financial_data)
    
    # Check for target correlation
    financial_corr = checker.check_target_correlation(
        financial_df, 
        label_col=args.label_col,
        output_file='financial_target_correlation.png'
    )
    
    # Check for outperformance correlation
    financial_outperf_corr = checker.check_outperformance_correlation(
        financial_df,
        output_file='financial_outperformance_correlation.png'
    )
    
    # Check for setup_id patterns
    checker.check_setup_id_patterns(financial_df, label_col=args.label_col)
    
    # Check feature importance vs correlation
    if args.financial_importance:
        checker.check_feature_importance_vs_correlation(
            args.financial_importance,
            financial_corr,
            output_file='financial_importance_vs_correlation.png'
        )
    
    # Check train-test overlap
    if args.text_test_data and args.financial_test_data:
        logger.info("\n=== Checking Train-Test Overlap ===")
        
        # Load test data
        text_test_df = checker.load_data(args.text_test_data)
        financial_test_df = checker.load_data(args.financial_test_data)
        
        # Check overlap
        checker.check_train_test_overlap(text_df, text_test_df)
        checker.check_train_test_overlap(financial_df, financial_test_df)
    
    logger.info("\n=== Analysis Complete ===")

if __name__ == '__main__':
    main() 