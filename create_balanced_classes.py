#!/usr/bin/env python3
"""
Create Balanced Classes

This script creates balanced classes for ML datasets by sorting outperformance values
and applying dynamic thresholds to achieve equal class distribution.

Usage:
    python create_balanced_classes.py --input data/ml_features/text_ml_features_training_balanced.csv 
                                     --output data/ml_features/text_ml_features_training_balanced_classes.csv
                                     --class-ratio 0.33,0.33,0.34
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

class BalancedClassCreator:
    """Class for creating balanced classes in ML datasets"""
    
    def __init__(
        self,
        output_dir: str = "data/ml_features/balanced",
        class_ratio: List[float] = None,
        save_thresholds: bool = True
    ):
        """
        Initialize the class creator
        
        Args:
            output_dir: Directory to save balanced datasets
            class_ratio: Ratio of classes (negative, neutral, positive)
            save_thresholds: Whether to save thresholds to file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Default to equal class distribution if not provided
        if class_ratio is None:
            self.class_ratio = [0.33, 0.33, 0.34]  # Negative, Neutral, Positive
        else:
            self.class_ratio = class_ratio
            
        # Validate class ratio
        if len(self.class_ratio) != 3 or abs(sum(self.class_ratio) - 1.0) > 0.01:
            raise ValueError("Class ratio must have 3 values that sum to 1.0")
            
        self.save_thresholds = save_thresholds
        self.thresholds = {}
    
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
    
    def create_balanced_classes(
        self,
        input_file: str,
        output_file: str = None,
        mode: str = "training"
    ) -> str:
        """
        Create balanced classes in ML dataset
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            mode: Mode ('training' or 'prediction')
            
        Returns:
            Path to output CSV file
        """
        # Load data
        df = self.load_data(input_file)
        
        # Check if outperformance_10d column exists
        if 'outperformance_10d' not in df.columns:
            logger.error(f"No 'outperformance_10d' column found in {input_file}")
            return None
        
        # For training mode, calculate thresholds and apply them
        if mode == "training":
            # Sort by outperformance
            sorted_outperf = df['outperformance_10d'].sort_values().reset_index(drop=True)
            
            # Calculate thresholds
            n_samples = len(sorted_outperf)
            neg_threshold_idx = int(n_samples * self.class_ratio[0])
            neutral_threshold_idx = int(n_samples * (self.class_ratio[0] + self.class_ratio[1]))
            
            neg_threshold = sorted_outperf[neg_threshold_idx]
            pos_threshold = sorted_outperf[neutral_threshold_idx]
            
            logger.info(f"Calculated thresholds: negative={neg_threshold:.4f}, positive={pos_threshold:.4f}")
            
            # Store thresholds
            self.thresholds = {
                'negative_threshold': neg_threshold,
                'positive_threshold': pos_threshold
            }
            
            # Apply thresholds to create balanced classes
            df['label'] = df['outperformance_10d'].apply(
                lambda x: 0 if x <= neg_threshold else (2 if x >= pos_threshold else 1)
            )
            
            # Save thresholds to file
            if self.save_thresholds:
                thresholds_file = self.output_dir / "class_thresholds.csv"
                pd.DataFrame({
                    'threshold_name': ['negative_threshold', 'positive_threshold'],
                    'value': [neg_threshold, pos_threshold],
                    'timestamp': [datetime.now().isoformat(), datetime.now().isoformat()]
                }).to_csv(thresholds_file, index=False)
                logger.info(f"Saved thresholds to {thresholds_file}")
        
        # For prediction mode, load thresholds and apply them
        else:
            # Load thresholds from file
            thresholds_file = self.output_dir / "class_thresholds.csv"
            if thresholds_file.exists():
                thresholds_df = pd.read_csv(thresholds_file)
                neg_threshold = thresholds_df.loc[thresholds_df['threshold_name'] == 'negative_threshold', 'value'].iloc[0]
                pos_threshold = thresholds_df.loc[thresholds_df['threshold_name'] == 'positive_threshold', 'value'].iloc[0]
                
                logger.info(f"Loaded thresholds: negative={neg_threshold:.4f}, positive={pos_threshold:.4f}")
                
                # Apply thresholds to create classes
                df['label'] = df['outperformance_10d'].apply(
                    lambda x: 0 if x <= neg_threshold else (2 if x >= pos_threshold else 1)
                )
            else:
                logger.warning(f"No thresholds file found at {thresholds_file}, using default thresholds")
                # Use default thresholds
                neg_threshold = -0.02
                pos_threshold = 0.02
                
                logger.info(f"Using default thresholds: negative={neg_threshold:.4f}, positive={pos_threshold:.4f}")
                
                # Apply thresholds to create classes
                df['label'] = df['outperformance_10d'].apply(
                    lambda x: 0 if x <= neg_threshold else (2 if x >= pos_threshold else 1)
                )
        
        # Log class distribution
        class_counts = df['label'].value_counts()
        logger.info("Class distribution after balancing:")
        for cls, count in class_counts.items():
            class_name = "Positive (2)" if cls == 2 else "Neutral (1)" if cls == 1 else "Negative (0)"
            logger.info(f"- Class {cls} ({class_name}): {count} ({count/len(df):.1%})")
        
        # Generate output file name if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_path = Path(input_file)
            output_file = self.output_dir / f"{input_path.stem}_balanced_classes_{timestamp}.csv"
        
        # Save balanced dataset
        df.to_csv(output_file, index=False)
        logger.info(f"Saved balanced classes dataset to {output_file}")
        
        return str(output_file)
    
    def process_multiple_files(
        self,
        files_dict: Dict[str, str],
        output_suffix: str = "balanced_classes"
    ) -> Dict[str, str]:
        """
        Process multiple files with the same thresholds
        
        Args:
            files_dict: Dictionary mapping file types to file paths
            output_suffix: Suffix to add to output file names
            
        Returns:
            Dictionary with paths to processed files
        """
        output_dict = {}
        
        # Process training files first to calculate thresholds
        for file_type, file_path in files_dict.items():
            if "train" in file_type:
                output_path = Path(file_path).parent / f"{Path(file_path).stem}_{output_suffix}.csv"
                output_dict[file_type] = self.create_balanced_classes(
                    file_path, str(output_path), "training"
                )
        
        # Then process prediction files using the same thresholds
        for file_type, file_path in files_dict.items():
            if "predict" in file_type:
                output_path = Path(file_path).parent / f"{Path(file_path).stem}_{output_suffix}.csv"
                output_dict[file_type] = self.create_balanced_classes(
                    file_path, str(output_path), "prediction"
                )
        
        return output_dict

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create balanced classes')
    parser.add_argument('--input', required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output',
                       help='Path to output CSV file')
    parser.add_argument('--output-dir', default='data/ml_features/balanced',
                       help='Directory to save balanced datasets')
    parser.add_argument('--class-ratio', default='0.33,0.33,0.34',
                       help='Ratio of classes (negative,neutral,positive)')
    parser.add_argument('--mode', choices=['training', 'prediction'], default='training',
                       help='Mode (training or prediction)')
    
    args = parser.parse_args()
    
    # Parse class ratio
    class_ratio = [float(r) for r in args.class_ratio.split(',')]
    
    # Initialize creator
    creator = BalancedClassCreator(
        output_dir=args.output_dir,
        class_ratio=class_ratio
    )
    
    # Create balanced classes
    creator.create_balanced_classes(
        input_file=args.input,
        output_file=args.output,
        mode=args.mode
    )

if __name__ == '__main__':
    main() 