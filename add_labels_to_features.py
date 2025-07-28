#!/usr/bin/env python3
"""
Add Labels to ML Feature Tables

This script adds labels from the DuckDB database to the ML feature tables.

Usage:
    python add_labels_to_features.py --input data/ml_features/text_ml_features_training_*.csv --output data/ml_features/text_ml_features_training_labeled.csv
"""

import os
import sys
import argparse
import logging
import duckdb
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

class LabelAdder:
    """Class for adding labels to ML feature tables"""
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        output_dir: str = "data/ml_features"
    ):
        """
        Initialize the label adder
        
        Args:
            db_path: Path to DuckDB database
            output_dir: Directory to save labeled features
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
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
        
        return df
    
    def get_labels_from_db(self, setup_ids: List[str], mode: str = 'training') -> pd.DataFrame:
        """
        Get labels from the database
        
        Args:
            setup_ids: List of setup IDs
            mode: Mode ('training' or 'prediction')
            
        Returns:
            DataFrame with labels
        """
        logger.info(f"Getting labels for {len(setup_ids)} setup IDs")
        
        conn = None
        try:
            conn = duckdb.connect(self.db_path)
            
            # For training mode, get labels from the labels table
            if mode == 'training':
                # Get average outperformance for first 10 days
                outperformance_df = conn.execute("""
                    SELECT 
                        setup_id, 
                        AVG(outperformance_day) as outperformance_10d
                    FROM daily_labels
                    WHERE day_number <= 10
                    AND setup_id = ANY(?)
                    GROUP BY setup_id
                    HAVING COUNT(DISTINCT day_number) >= 2
                """, [setup_ids]).df()
                
                if len(outperformance_df) > 0:
                    # Calculate percentile thresholds for balanced classes
                    neg_threshold = np.percentile(outperformance_df['outperformance_10d'], 33.33)
                    pos_threshold = np.percentile(outperformance_df['outperformance_10d'], 66.67)
                    
                    logger.info(f"Using dynamic thresholds for balanced classes:")
                    logger.info(f"- Negative threshold (33.33%): {neg_threshold:.4f}")
                    logger.info(f"- Positive threshold (66.67%): {pos_threshold:.4f}")
                    
                    # Add label_class based on percentile thresholds
                    outperformance_df['label_class'] = outperformance_df['outperformance_10d'].apply(
                        lambda x: 1 if x >= pos_threshold else (-1 if x <= neg_threshold else 0)
                    )
                    
                    labels_df = outperformance_df
                else:
                    logger.warning("No outperformance data found, using empty DataFrame")
                    labels_df = pd.DataFrame(columns=['setup_id', 'outperformance_10d', 'label_class'])
                
                logger.info(f"Found {len(labels_df)} labels")
                
                # Log class distribution
                if len(labels_df) > 0:
                    class_counts = labels_df['label_class'].value_counts()
                    logger.info("Class distribution:")
                    for cls, count in class_counts.items():
                        class_name = "Positive (1)" if cls == 1 else "Neutral (0)" if cls == 0 else "Negative (-1)"
                        logger.info(f"- Class {cls} ({class_name}): {count} ({count/len(labels_df):.1%})")
            
            # For prediction mode, we don't have labels, but we can still get outperformance if available
            else:
                # Try to get outperformance from the database
                outperformance_df = conn.execute("""
                    SELECT 
                        setup_id, 
                        AVG(outperformance_day) as outperformance_10d
                    FROM daily_labels
                    WHERE day_number <= 10
                    AND setup_id = ANY(?)
                    GROUP BY setup_id
                """, [setup_ids]).df()
                
                if len(outperformance_df) > 0:
                    # Use the same thresholds as for training
                    neg_threshold = -0.21  # Approximately 33.33 percentile
                    pos_threshold = 0.28   # Approximately 66.67 percentile
                    
                    logger.info(f"Using fixed thresholds for prediction data:")
                    logger.info(f"- Negative threshold: {neg_threshold:.4f}")
                    logger.info(f"- Positive threshold: {pos_threshold:.4f}")
                    
                    # Add label_class based on fixed thresholds
                    outperformance_df['label_class'] = outperformance_df['outperformance_10d'].apply(
                        lambda x: 1 if x >= pos_threshold else (-1 if x <= neg_threshold else 0)
                    )
                    
                    labels_df = outperformance_df
                else:
                    logger.warning("No outperformance data found for prediction, using empty DataFrame")
                    labels_df = pd.DataFrame(columns=['setup_id', 'outperformance_10d', 'label_class'])
                
                logger.info(f"Found {len(labels_df)} labels for prediction data (for evaluation only)")
            
            return labels_df
        
        finally:
            if conn:
                conn.close()
    
    def add_labels(self, df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add labels to ML features
        
        Args:
            df: DataFrame with ML features
            labels_df: DataFrame with labels
            
        Returns:
            DataFrame with ML features and labels
        """
        logger.info("Adding labels to ML features")
        
        # Merge features with labels
        df_labeled = pd.merge(df, labels_df, on='setup_id', how='left')
        
        # Count rows with labels
        labeled_rows = df_labeled['label_class'].notna().sum()
        logger.info(f"Rows with labels: {labeled_rows}/{len(df_labeled)} ({labeled_rows/len(df_labeled):.1%})")
        
        # Rename label_class to label
        if 'label' in df_labeled.columns:
            # If label column already exists, check if it's empty
            if df_labeled['label'].isna().all():
                # Replace it with label_class
                df_labeled['label'] = df_labeled['label_class']
                df_labeled = df_labeled.drop(columns=['label_class'])
                logger.info("Replaced empty 'label' column with 'label_class'")
            else:
                # Keep both columns
                logger.info("Both 'label' and 'label_class' columns exist and have values")
        else:
            # Rename label_class to label
            df_labeled = df_labeled.rename(columns={'label_class': 'label'})
            logger.info("Renamed 'label_class' to 'label'")
        
        return df_labeled
    
    def process_file(self, input_file: str, output_file: str = None, mode: str = 'training') -> str:
        """
        Process a ML features file
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            mode: Mode ('training' or 'prediction')
            
        Returns:
            Path to output CSV file
        """
        # Load data
        df = self.load_data(input_file)
        
        # Get setup IDs
        setup_ids = df['setup_id'].tolist()
        
        # Get labels from database
        labels_df = self.get_labels_from_db(setup_ids, mode)
        
        # Add labels to features
        df_labeled = self.add_labels(df, labels_df)
        
        # Generate output file name if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_path = Path(input_file)
            output_file = self.output_dir / f"labeled_{input_path.name.replace('.csv', '')}_{timestamp}.csv"
        
        # Save labeled features
        df_labeled.to_csv(output_file, index=False)
        logger.info(f"Saved labeled features to {output_file}")
        
        # Print summary
        logger.info("\nLabeled Features Summary:")
        logger.info(f"- Input rows: {len(df)}")
        logger.info(f"- Output rows: {len(df_labeled)}")
        logger.info(f"- Rows with labels: {df_labeled['label'].notna().sum()}")
        
        # Print class distribution
        if df_labeled['label'].notna().sum() > 0:
            class_counts = df_labeled['label'].value_counts()
            logger.info("\nClass distribution:")
            for cls, count in class_counts.items():
                logger.info(f"- Class {cls}: {count} ({count/len(df_labeled):.1%})")
        
        return str(output_file)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Add labels to ML feature tables')
    parser.add_argument('--input', required=True,
                       help='Path to input ML features CSV')
    parser.add_argument('--output',
                       help='Path to output labeled features CSV')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--output-dir', default='data/ml_features',
                       help='Directory to save labeled features')
    parser.add_argument('--mode', choices=['training', 'prediction'], default='training',
                       help='Mode (training or prediction)')
    
    args = parser.parse_args()
    
    # Initialize label adder
    label_adder = LabelAdder(
        db_path=args.db_path,
        output_dir=args.output_dir
    )
    
    # Process file
    label_adder.process_file(
        input_file=args.input,
        output_file=args.output,
        mode=args.mode
    )

if __name__ == '__main__':
    main() 