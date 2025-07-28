#!/usr/bin/env python3
"""
Fix Data Leakage in ML Pipeline

This script fixes data leakage issues by:
1. Creating a proper train-test split with no overlap
2. Removing target leakage (outperformance_10d)
3. Implementing proper data preprocessing

Usage:
    python fix_data_leakage.py --input-text data/ml_features/text_ml_features_training_labeled.csv 
                              --input-financial data/ml_features/financial_ml_features_training_labeled.csv 
                              --output-dir data/ml_features_clean
                              --test-size 0.2
                              --time-split
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLeakageFixer:
    """Class for fixing data leakage issues in ML pipeline"""
    
    def __init__(self, output_dir: str = "data/ml_features_clean"):
        """
        Initialize the fixer
        
        Args:
            output_dir: Directory to save clean data
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
    
    def extract_date_from_setup_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract date from setup_id
        
        Args:
            df: DataFrame with setup_id column
            
        Returns:
            DataFrame with date column
        """
        logger.info("Extracting date from setup_id")
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Extract date part from setup_id (assuming format like "BBB_2024-04-29")
        result_df['ticker'] = result_df['setup_id'].str.split('_', expand=True)[0]
        
        try:
            # Extract date if available
            result_df['date_str'] = result_df['setup_id'].str.split('_', expand=True)[1]
            
            # Convert to datetime for those with valid date format
            valid_date_mask = result_df['date_str'].str.match(r'\d{4}-\d{2}-\d{2}')
            if valid_date_mask.any():
                result_df.loc[valid_date_mask, 'date'] = pd.to_datetime(
                    result_df.loc[valid_date_mask, 'date_str']
                )
            
            # For rows without valid date, set to NaT
            result_df.loc[~valid_date_mask, 'date'] = pd.NaT
            
            # Drop temporary column
            result_df = result_df.drop(columns=['date_str'])
            
            # Log info about valid dates
            valid_dates = result_df['date'].notna().sum()
            logger.info(f"Extracted {valid_dates} valid dates out of {len(result_df)} rows")
            
        except Exception as e:
            logger.warning(f"Could not extract dates from setup_id: {e}")
            result_df['date'] = pd.NaT
        
        return result_df
    
    def remove_target_leakage(self, df: pd.DataFrame, target_cols: List[str] = None) -> pd.DataFrame:
        """
        Remove columns that could cause target leakage
        
        Args:
            df: DataFrame with features
            target_cols: List of columns to remove
            
        Returns:
            DataFrame without target leakage columns
        """
        if target_cols is None:
            target_cols = ['outperformance_10d']
        
        logger.info(f"Removing target leakage columns: {target_cols}")
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Remove target leakage columns if they exist
        cols_to_remove = [col for col in target_cols if col in result_df.columns]
        if cols_to_remove:
            result_df = result_df.drop(columns=cols_to_remove)
            logger.info(f"Removed {len(cols_to_remove)} target leakage columns")
        else:
            logger.info("No target leakage columns found")
        
        return result_df
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2, 
        time_split: bool = True,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and test sets
        
        Args:
            df: DataFrame with features and labels
            test_size: Fraction of data to use for test set
            time_split: Whether to split based on time
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(f"Splitting data with test_size={test_size}, time_split={time_split}")
        
        # Check if we have date column for time-based split
        if time_split and 'date' in df.columns and df['date'].notna().any():
            # Sort by date
            df = df.sort_values('date')
            
            # Calculate split point
            split_idx = int(len(df) * (1 - test_size))
            
            # Split data
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            
            # Log split info
            if 'date' in train_df.columns and train_df['date'].notna().any():
                logger.info(f"Training data from {train_df['date'].min()} to {train_df['date'].max()}")
            if 'date' in test_df.columns and test_df['date'].notna().any():
                logger.info(f"Test data from {test_df['date'].min()} to {test_df['date'].max()}")
        else:
            # Fallback to random split if time-based split is not possible
            if time_split:
                logger.warning("Time-based split not possible, falling back to random split")
            
            # Get unique setup_ids to ensure no overlap
            setup_ids = df['setup_id'].unique()
            
            # Split setup_ids
            train_ids, test_ids = train_test_split(
                setup_ids, 
                test_size=test_size, 
                random_state=random_state
            )
            
            # Split data based on setup_ids
            train_df = df[df['setup_id'].isin(train_ids)].copy()
            test_df = df[df['setup_id'].isin(test_ids)].copy()
        
        logger.info(f"Split data into {len(train_df)} training samples and {len(test_df)} test samples")
        
        # Verify no overlap
        train_ids = set(train_df['setup_id'])
        test_ids = set(test_df['setup_id'])
        overlap = train_ids.intersection(test_ids)
        
        if overlap:
            logger.warning(f"Found {len(overlap)} overlapping setup_ids between training and test data")
            logger.warning(f"Examples: {list(overlap)[:5]}")
        else:
            logger.info("No overlap between training and test data")
        
        return train_df, test_df
    
    def preprocess_features(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        numeric_strategy: str = 'median',
        categorical_strategy: str = 'most_frequent'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess features (impute missing values, scale numeric features)
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            numeric_strategy: Strategy for imputing numeric features
            categorical_strategy: Strategy for imputing categorical features
            
        Returns:
            Tuple of (preprocessed_train_df, preprocessed_test_df)
        """
        logger.info("Preprocessing features")
        
        # Create copies to avoid modifying the originals
        train_result = train_df.copy()
        test_result = test_df.copy()
        
        # Remove non-feature columns
        exclude_cols = ['setup_id', 'ticker', 'date']
        for col in exclude_cols:
            if col in train_result.columns:
                train_result = train_result.drop(columns=[col])
            if col in test_result.columns:
                test_result = test_result.drop(columns=[col])
        
        # Identify numeric and categorical columns (excluding label)
        label_col = 'label'
        feature_cols = [col for col in train_result.columns if col != label_col]
        
        numeric_cols = train_result[feature_cols].select_dtypes(include=['number']).columns.tolist()
        categorical_cols = train_result[feature_cols].select_dtypes(exclude=['number']).columns.tolist()
        
        logger.info(f"Found {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns")
        
        # Impute missing values for numeric features
        if numeric_cols:
            numeric_imputer = SimpleImputer(strategy=numeric_strategy)
            train_result[numeric_cols] = numeric_imputer.fit_transform(train_result[numeric_cols])
            test_result[numeric_cols] = numeric_imputer.transform(test_result[numeric_cols])
            
            # Scale numeric features
            scaler = StandardScaler()
            train_result[numeric_cols] = scaler.fit_transform(train_result[numeric_cols])
            test_result[numeric_cols] = scaler.transform(test_result[numeric_cols])
            
            logger.info(f"Imputed and scaled {len(numeric_cols)} numeric columns")
        
        # Impute missing values for categorical features
        if categorical_cols:
            categorical_imputer = SimpleImputer(strategy=categorical_strategy)
            train_result[categorical_cols] = categorical_imputer.fit_transform(train_result[categorical_cols])
            test_result[categorical_cols] = categorical_imputer.transform(test_result[categorical_cols])
            
            logger.info(f"Imputed {len(categorical_cols)} categorical columns")
        
        return train_result, test_result
    
    def process_data(
        self,
        input_file: str,
        output_dir: str = None,
        test_size: float = 0.2,
        time_split: bool = True,
        random_state: int = 42
    ) -> Tuple[str, str]:
        """
        Process data to fix leakage issues
        
        Args:
            input_file: Path to input CSV file
            output_dir: Directory to save output files
            test_size: Fraction of data to use for test set
            time_split: Whether to split based on time
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (train_file, test_file) paths
        """
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get base filename without extension
        base_name = Path(input_file).stem
        
        # Load data
        df = self.load_data(input_file)
        
        # Extract date from setup_id
        df = self.extract_date_from_setup_id(df)
        
        # Remove target leakage
        df = self.remove_target_leakage(df)
        
        # Split data
        train_df, test_df = self.split_data(
            df, 
            test_size=test_size, 
            time_split=time_split,
            random_state=random_state
        )
        
        # Preprocess features
        train_df, test_df = self.preprocess_features(train_df, test_df)
        
        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_file = output_dir / f"{base_name}_train_clean_{timestamp}.csv"
        test_file = output_dir / f"{base_name}_test_clean_{timestamp}.csv"
        
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        logger.info(f"Saved training data to {train_file}")
        logger.info(f"Saved test data to {test_file}")
        
        return str(train_file), str(test_file)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fix data leakage in ML pipeline')
    parser.add_argument('--input-text', required=True,
                      help='Path to text ML features CSV')
    parser.add_argument('--input-financial', required=True,
                      help='Path to financial ML features CSV')
    parser.add_argument('--output-dir', default='data/ml_features_clean',
                      help='Directory to save clean data')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Fraction of data to use for test set')
    parser.add_argument('--time-split', action='store_true',
                      help='Whether to split based on time')
    parser.add_argument('--random-state', type=int, default=42,
                      help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize fixer
    fixer = DataLeakageFixer(output_dir=args.output_dir)
    
    # Process text data
    logger.info("=== Processing Text Features ===")
    text_train_file, text_test_file = fixer.process_data(
        input_file=args.input_text,
        output_dir=args.output_dir,
        test_size=args.test_size,
        time_split=args.time_split,
        random_state=args.random_state
    )
    
    # Process financial data
    logger.info("\n=== Processing Financial Features ===")
    financial_train_file, financial_test_file = fixer.process_data(
        input_file=args.input_financial,
        output_dir=args.output_dir,
        test_size=args.test_size,
        time_split=args.time_split,
        random_state=args.random_state
    )
    
    logger.info("\n=== Processing Complete ===")
    logger.info(f"Text training data: {text_train_file}")
    logger.info(f"Text test data: {text_test_file}")
    logger.info(f"Financial training data: {financial_train_file}")
    logger.info(f"Financial test data: {financial_test_file}")

if __name__ == '__main__':
    main() 