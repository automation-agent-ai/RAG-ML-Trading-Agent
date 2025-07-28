#!/usr/bin/env python3
"""
Balance ML Datasets

This script ensures that text and financial ML datasets have the same setup_ids
and sample sizes for both training and prediction datasets.

Usage:
    python balance_ml_datasets.py --text-train data/ml_features/text_ml_features_training_labeled.csv 
                                 --financial-train data/ml_features/financial_ml_features_training_labeled.csv
                                 --text-predict data/ml_features/text_ml_features_prediction_labeled.csv
                                 --financial-predict data/ml_features/financial_ml_features_prediction_labeled.csv
                                 --output-dir data/ml_features/balanced
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLDatasetBalancer:
    """Class for balancing text and financial ML datasets"""
    
    def __init__(
        self,
        output_dir: str = "data/ml_features/balanced",
        target_size: int = 0,  # 0 means use all available setup_ids
        random_seed: int = 42
    ):
        """
        Initialize the balancer
        
        Args:
            output_dir: Directory to save balanced datasets
            target_size: Target number of setup_ids for each dataset
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.target_size = target_size
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load ML features from CSV
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with ML features
        """
        logger.info(f"Loading ML features from {file_path}")
        df = pd.read_csv(file_path)
        
        # Basic info
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Found {df['setup_id'].nunique()} unique setup_ids")
        
        return df
    
    def _convert_label_to_consistent_format(self, label, target_format='numeric'):
        """
        Convert label to consistent format
        
        Args:
            label: The label to convert
            target_format: 'numeric' for -1/0/1 or 'string' for negative/neutral/positive
            
        Returns:
            Converted label
        """
        # Convert to numeric format (-1, 0, 1)
        if target_format == 'numeric':
            if isinstance(label, str):
                if label.lower() == 'positive':
                    return 1
                elif label.lower() == 'negative':
                    return -1
                elif label.lower() == 'neutral':
                    return 0
            # Handle old numeric format (0, 1, 2)
            elif isinstance(label, (int, float)):
                if label == 2.0 or label == 2:
                    return 1  # Convert old 2 to new 1
                elif label == 0.0 or label == 0:
                    return -1  # Convert old 0 to new -1
                elif label == 1.0 or label == 1:
                    return 0  # Convert old 1 to new 0
            return label
        # Convert to string format (negative, neutral, positive)
        elif target_format == 'string':
            if isinstance(label, (int, float)):
                if label == 1 or label == 1.0:
                    return 'positive'
                elif label == -1 or label == -1.0:
                    return 'negative'
                elif label == 0 or label == 0.0:
                    return 'neutral'
                # Handle old numeric format
                elif label == 2 or label == 2.0:
                    return 'positive'
            return label
        return label
    
    def _get_label_format(self, df):
        """
        Determine the label format in a dataframe
        
        Args:
            df: DataFrame with labels
            
        Returns:
            'numeric' or 'string'
        """
        if 'label' not in df.columns:
            return 'unknown'
        
        # Get a sample label
        sample_labels = df['label'].dropna().unique()
        if len(sample_labels) == 0:
            return 'unknown'
        
        sample_label = sample_labels[0]
        if isinstance(sample_label, (int, float)):
            return 'numeric'
        elif isinstance(sample_label, str):
            return 'string'
        return 'unknown'
    
    def balance_datasets(
        self,
        text_train_file: str,
        financial_train_file: str,
        text_predict_file: str,
        financial_predict_file: str
    ) -> Dict[str, str]:
        """
        Balance text and financial ML datasets
        
        Args:
            text_train_file: Path to text training CSV
            financial_train_file: Path to financial training CSV
            text_predict_file: Path to text prediction CSV
            financial_predict_file: Path to financial prediction CSV
            
        Returns:
            Dictionary with paths to balanced datasets
        """
        # Load datasets
        text_train = self.load_data(text_train_file)
        financial_train = self.load_data(financial_train_file)
        text_predict = self.load_data(text_predict_file)
        financial_predict = self.load_data(financial_predict_file)
        
        # Determine label formats
        text_label_format = self._get_label_format(text_train)
        financial_label_format = self._get_label_format(financial_train)
        logger.info(f"Text label format: {text_label_format}")
        logger.info(f"Financial label format: {financial_label_format}")
        
        # Balance training datasets
        logger.info("Balancing training datasets")
        balanced_text_train, balanced_financial_train = self._balance_pair(
            text_train, financial_train, "training", 
            text_label_format, financial_label_format
        )
        
        # Balance prediction datasets
        logger.info("Balancing prediction datasets")
        balanced_text_predict, balanced_financial_predict = self._balance_pair(
            text_predict, financial_predict, "prediction",
            text_label_format, financial_label_format
        )
        
        # Save balanced datasets
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        text_train_output = self.output_dir / f"text_ml_features_training_balanced_{timestamp}.csv"
        financial_train_output = self.output_dir / f"financial_ml_features_training_balanced_{timestamp}.csv"
        text_predict_output = self.output_dir / f"text_ml_features_prediction_balanced_{timestamp}.csv"
        financial_predict_output = self.output_dir / f"financial_ml_features_prediction_balanced_{timestamp}.csv"
        
        balanced_text_train.to_csv(text_train_output, index=False)
        balanced_financial_train.to_csv(financial_train_output, index=False)
        balanced_text_predict.to_csv(text_predict_output, index=False)
        balanced_financial_predict.to_csv(financial_predict_output, index=False)
        
        logger.info(f"Saved balanced text training dataset to {text_train_output}")
        logger.info(f"Saved balanced financial training dataset to {financial_train_output}")
        logger.info(f"Saved balanced text prediction dataset to {text_predict_output}")
        logger.info(f"Saved balanced financial prediction dataset to {financial_predict_output}")
        
        return {
            "text_train": str(text_train_output),
            "financial_train": str(financial_train_output),
            "text_predict": str(text_predict_output),
            "financial_predict": str(financial_predict_output)
        }
    
    def _balance_pair(
        self,
        text_df: pd.DataFrame,
        financial_df: pd.DataFrame,
        mode: str,
        text_label_format: str = 'string',
        financial_label_format: str = 'numeric'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Balance a pair of text and financial datasets
        
        Args:
            text_df: Text ML features DataFrame
            financial_df: Financial ML features DataFrame
            mode: Mode ('training' or 'prediction')
            text_label_format: Format of labels in text_df ('numeric' or 'string')
            financial_label_format: Format of labels in financial_df ('numeric' or 'string')
            
        Returns:
            Tuple of balanced text and financial DataFrames
        """
        # Get setup_ids
        text_setup_ids = set(text_df['setup_id'].unique())
        financial_setup_ids = set(financial_df['setup_id'].unique())
        
        # Find common setup_ids
        common_setup_ids = text_setup_ids.intersection(financial_setup_ids)
        logger.info(f"Found {len(common_setup_ids)} common setup_ids")
        
        # If target_size is 0, use all available setup_ids from text dataset
        if self.target_size == 0:
            logger.info(f"Using all {len(text_setup_ids)} setup_ids from text dataset")
            selected_setup_ids = list(text_setup_ids)
            
            # Add any missing setup_ids to financial_df
            missing_in_financial = text_setup_ids - financial_setup_ids
            if missing_in_financial:
                logger.info(f"Adding {len(missing_in_financial)} setup_ids to financial dataset")
                for setup_id in missing_in_financial:
                    # Create a dummy row by copying the first row and changing setup_id
                    dummy_row = financial_df.iloc[0:1].copy()
                    dummy_row['setup_id'] = setup_id
                    # Copy the label from text_df to ensure consistency
                    if 'label' in text_df.columns and 'label' in dummy_row.columns:
                        label_value = text_df[text_df['setup_id'] == setup_id]['label'].values[0]
                        # Convert label to financial format if needed
                        dummy_row['label'] = self._convert_label_to_consistent_format(
                            label_value, 
                            target_format=financial_label_format
                        )
                    financial_df = pd.concat([financial_df, dummy_row], ignore_index=True)
        # If we have a specific target size
        elif len(common_setup_ids) >= self.target_size:
            logger.info(f"Using {self.target_size} common setup_ids")
            # Randomly select target_size setup_ids
            selected_setup_ids = self.rng.choice(
                list(common_setup_ids),
                size=self.target_size,
                replace=False
            )
        else:
            # Use all common setup_ids
            selected_setup_ids = list(common_setup_ids)
            
            # If we need more setup_ids, add from each dataset
            remaining = self.target_size - len(selected_setup_ids)
            if remaining > 0:
                logger.info(f"Need {remaining} more setup_ids to reach target size")
                
                # Get setup_ids unique to each dataset
                text_only = text_setup_ids - common_setup_ids
                financial_only = financial_setup_ids - common_setup_ids
                
                # Add setup_ids from both datasets
                text_add = min(remaining // 2, len(text_only))
                financial_add = min(remaining - text_add, len(financial_only))
                
                if text_add > 0:
                    text_additional = self.rng.choice(list(text_only), size=text_add, replace=False)
                    selected_setup_ids.extend(text_additional)
                    
                    # Add dummy rows to financial_df for these setup_ids
                    for setup_id in text_additional:
                        # Create a dummy row by copying the first row and changing setup_id
                        dummy_row = financial_df.iloc[0:1].copy()
                        dummy_row['setup_id'] = setup_id
                        # Copy the label from text_df to ensure consistency
                        if 'label' in text_df.columns and 'label' in dummy_row.columns:
                            label_value = text_df[text_df['setup_id'] == setup_id]['label'].values[0]
                            # Convert label to financial format if needed
                            dummy_row['label'] = self._convert_label_to_consistent_format(
                                label_value, 
                                target_format=financial_label_format
                            )
                        financial_df = pd.concat([financial_df, dummy_row], ignore_index=True)
                
                if financial_add > 0:
                    financial_additional = self.rng.choice(list(financial_only), size=financial_add, replace=False)
                    selected_setup_ids.extend(financial_additional)
                    
                    # Add dummy rows to text_df for these setup_ids
                    for setup_id in financial_additional:
                        # Create a dummy row by copying the first row and changing setup_id
                        dummy_row = text_df.iloc[0:1].copy()
                        dummy_row['setup_id'] = setup_id
                        # Copy the label from financial_df to ensure consistency
                        if 'label' in financial_df.columns and 'label' in dummy_row.columns:
                            label_value = financial_df[financial_df['setup_id'] == setup_id]['label'].values[0]
                            # Convert label to text format if needed
                            dummy_row['label'] = self._convert_label_to_consistent_format(
                                label_value, 
                                target_format=text_label_format
                            )
                        text_df = pd.concat([text_df, dummy_row], ignore_index=True)
        
        # Filter datasets to selected setup_ids
        balanced_text_df = text_df[text_df['setup_id'].isin(selected_setup_ids)]
        balanced_financial_df = financial_df[financial_df['setup_id'].isin(selected_setup_ids)]
        
        # Ensure label consistency across both datasets
        if 'label' in balanced_text_df.columns and 'label' in balanced_financial_df.columns:
            # Create a mapping of setup_id to consistent label
            label_mapping = {}
            for setup_id in selected_setup_ids:
                text_row = balanced_text_df[balanced_text_df['setup_id'] == setup_id]
                financial_row = balanced_financial_df[balanced_financial_df['setup_id'] == setup_id]
                
                if len(text_row) > 0 and len(financial_row) > 0:
                    text_label = text_row['label'].values[0]
                    financial_label = financial_row['label'].values[0]
                    
                    # If labels don't match after conversion, use text label as the source of truth
                    text_label_numeric = self._convert_label_to_consistent_format(text_label, 'numeric')
                    financial_label_numeric = self._convert_label_to_consistent_format(financial_label, 'numeric')
                    
                    if text_label_numeric != financial_label_numeric:
                        # Update financial label to match text label
                        balanced_financial_df.loc[balanced_financial_df['setup_id'] == setup_id, 'label'] = self._convert_label_to_consistent_format(
                            text_label, 
                            target_format=financial_label_format
                        )
        
        # Check class distribution
        if 'label' in balanced_text_df.columns and 'label' in balanced_financial_df.columns:
            logger.info("Class distribution in balanced text dataset:")
            text_class_counts = balanced_text_df['label'].value_counts()
            for cls, count in text_class_counts.items():
                logger.info(f"- Class {cls}: {count} ({count/len(balanced_text_df):.1%})")
            
            logger.info("Class distribution in balanced financial dataset:")
            financial_class_counts = balanced_financial_df['label'].value_counts()
            for cls, count in financial_class_counts.items():
                logger.info(f"- Class {cls}: {count} ({count/len(balanced_financial_df):.1%})")
        
        return balanced_text_df, balanced_financial_df

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Balance ML datasets')
    parser.add_argument('--text-train', required=True,
                       help='Path to text training CSV')
    parser.add_argument('--financial-train', required=True,
                       help='Path to financial training CSV')
    parser.add_argument('--text-predict', required=True,
                       help='Path to text prediction CSV')
    parser.add_argument('--financial-predict', required=True,
                       help='Path to financial prediction CSV')
    parser.add_argument('--output-dir', default='data/ml_features/balanced',
                       help='Directory to save balanced datasets')
    parser.add_argument('--target-size', type=int, default=0,
                       help='Target number of setup_ids for each dataset (0 means use all available)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize balancer
    balancer = MLDatasetBalancer(
        output_dir=args.output_dir,
        target_size=args.target_size,
        random_seed=args.random_seed
    )
    
    # Balance datasets
    balancer.balance_datasets(
        text_train_file=args.text_train,
        financial_train_file=args.financial_train,
        text_predict_file=args.text_predict,
        financial_predict_file=args.financial_predict
    )

if __name__ == '__main__':
    main() 