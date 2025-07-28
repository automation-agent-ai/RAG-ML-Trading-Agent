#!/usr/bin/env python3
"""
Corrected CSV Feature Loader for ML Pipeline

Loads text and financial features from prepared CSV files in data/ml_features/balanced/
Properly handles the existing label column with values -1, 0, 1
"""

import numpy as np
import pandas as pd
import glob
import os
from typing import List, Optional, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorrectedCSVFeatureLoader:
    """Feature loader that uses prepared CSV files with correct label handling"""
    
    def __init__(self, data_dir: str = "data/ml_features/balanced"):
        self.data_dir = data_dir
        
    def load_text_features(self, mode: str = "training") -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load text features from CSV files
        
        Args:
            mode: Either "training" or "prediction"
            
        Returns:
            Tuple of (features_df, labels_array)
        """
        # Find the correct CSV file
        pattern = f"{self.data_dir}/text_ml_features_{mode}_balanced_*.csv"
        csv_files = glob.glob(pattern)
        
        if not csv_files:
            # Try a more flexible pattern as fallback
            pattern = f"{self.data_dir}/text_ml_features_{mode}*.csv"
            csv_files = glob.glob(pattern)
            
        if not csv_files:
            logger.error(f"No text {mode} CSV files found matching pattern: {pattern}")
            return None, None
            
        # Use the most recent file
        csv_file = sorted(csv_files)[-1]
        logger.info(f"Loading text {mode} features from: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded text {mode} data shape: {df.shape}")
            
            # Separate features and labels
            if 'setup_id' in df.columns:
                df = df.set_index('setup_id')
                
            # Extract labels (prioritize 'label' column if it exists)
            if 'label' in df.columns:
                labels = df['label'].values
                # Convert -1, 0, 1 to 0, 1, 2 for sklearn compatibility
                labels = labels + 1  # -1->0, 0->1, 1->2
                feature_cols = [col for col in df.columns 
                              if col not in ['label', 'outperformance_10d']]
                features = df[feature_cols].copy()
                
            elif 'outperformance_10d' in df.columns:
                # Convert to class labels
                labels = df['outperformance_10d'].apply(lambda x: 
                    2 if x > 5 else (0 if x < -5 else 1)  # positive, negative, neutral
                ).values
                
                # Remove target and other non-feature columns
                feature_cols = [col for col in df.columns 
                              if col not in ['outperformance_10d', 'label']]
                features = df[feature_cols].copy()
            else:
                logger.warning(f"No label column found in {mode} data")
                labels = None
                features = df.copy()
                
            # Ensure all features are numeric
            for col in features.columns:
                features.loc[:, col] = pd.to_numeric(features[col], errors='coerce')
                
            features = features.fillna(0)
            
            logger.info(f"Text {mode} features shape: {features.shape}")
            if labels is not None:
                logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
                
            return features, labels
            
        except Exception as e:
            logger.error(f"Error loading text {mode} features: {e}")
            return None, None
            
    def load_financial_features(self, mode: str = "training") -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load financial features from CSV files
        
        Args:
            mode: Either "training" or "prediction"
            
        Returns:
            Tuple of (features_df, labels_array)
        """
        # Find the correct CSV file
        pattern = f"{self.data_dir}/financial_ml_features_{mode}_balanced_*.csv"
        csv_files = glob.glob(pattern)
        
        if not csv_files:
            # Try a more flexible pattern as fallback
            pattern = f"{self.data_dir}/financial_ml_features_{mode}*.csv"
            csv_files = glob.glob(pattern)
            
        if not csv_files:
            logger.error(f"No financial {mode} CSV files found matching pattern: {pattern}")
            return None, None
            
        # Use the most recent file
        csv_file = sorted(csv_files)[-1]
        logger.info(f"Loading financial {mode} features from: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded financial {mode} data shape: {df.shape}")
            
            # Separate features and labels
            if 'setup_id' in df.columns:
                df = df.set_index('setup_id')
                
            # Extract labels (prioritize 'label' column if it exists)
            if 'label' in df.columns:
                labels = df['label'].values
                # Convert -1, 0, 1 to 0, 1, 2 for sklearn compatibility
                labels = labels + 1  # -1->0, 0->1, 1->2
                feature_cols = [col for col in df.columns 
                              if col not in ['label', 'outperformance_10d']]
                features = df[feature_cols].copy()
                
            elif 'outperformance_10d' in df.columns:
                # Convert to class labels
                labels = df['outperformance_10d'].apply(lambda x: 
                    2 if x > 5 else (0 if x < -5 else 1)  # positive, negative, neutral
                ).values
                
                # Remove target and other non-feature columns
                feature_cols = [col for col in df.columns 
                              if col not in ['outperformance_10d', 'label']]
                features = df[feature_cols].copy()
            else:
                logger.warning(f"No label column found in {mode} data")
                labels = None
                features = df.copy()
                
            # Ensure all features are numeric and handle NaN values
            for col in features.columns:
                features.loc[:, col] = pd.to_numeric(features[col], errors='coerce')
                
            features = features.fillna(0)
            
            # Clip extreme values for financial data
            features = features.clip(-1000, 1000)
            
            logger.info(f"Financial {mode} features shape: {features.shape}")
            if labels is not None:
                logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
                
            return features, labels
            
        except Exception as e:
            logger.error(f"Error loading financial {mode} features: {e}")
            return None, None
            
    def get_common_setup_ids(self) -> List[str]:
        """Get setup IDs that exist in both text and financial training data"""
        
        # Load both training datasets
        text_features, _ = self.load_text_features("training")
        financial_features, _ = self.load_financial_features("training")
        
        if text_features is None or financial_features is None:
            return []
            
        # Find common setup IDs
        text_ids = set(text_features.index)
        financial_ids = set(financial_features.index)
        common_ids = list(text_ids & financial_ids)
        
        logger.info(f"Found {len(common_ids)} common setup IDs between text and financial data")
        logger.info(f"Text training IDs: {len(text_ids)}")
        logger.info(f"Financial training IDs: {len(financial_ids)}")
        
        return common_ids
        
    def verify_data_consistency(self):
        """Verify that text and financial data have matching setup IDs"""
        
        # Check training data
        text_train, text_labels = self.load_text_features("training")
        financial_train, financial_labels = self.load_financial_features("training")
        
        if text_train is not None and financial_train is not None:
            text_ids = set(text_train.index)
            financial_ids = set(financial_train.index)
            
            if text_ids == financial_ids:
                logger.info("✅ Training data: Text and financial setup IDs match perfectly!")
            else:
                logger.warning(f"⚠️ Training data: {len(text_ids - financial_ids)} text-only IDs, {len(financial_ids - text_ids)} financial-only IDs")
                
            # Check label consistency
            if text_labels is not None and financial_labels is not None:
                common_ids = list(text_ids & financial_ids)
                text_labels_subset = text_labels[text_train.index.isin(common_ids)]
                financial_labels_subset = financial_labels[financial_train.index.isin(common_ids)]
                
                if np.array_equal(text_labels_subset, financial_labels_subset):
                    logger.info("✅ Training labels: Text and financial labels match perfectly!")
                else:
                    logger.warning("⚠️ Training labels: Text and financial labels differ")
        
        # Check prediction data
        text_pred, _ = self.load_text_features("prediction")
        financial_pred, _ = self.load_financial_features("prediction")
        
        if text_pred is not None and financial_pred is not None:
            text_pred_ids = set(text_pred.index)
            financial_pred_ids = set(financial_pred.index)
            
            if text_pred_ids == financial_pred_ids:
                logger.info("✅ Prediction data: Text and financial setup IDs match perfectly!")
            else:
                logger.warning(f"⚠️ Prediction data: {len(text_pred_ids - financial_pred_ids)} text-only IDs, {len(financial_pred_ids - text_pred_ids)} financial-only IDs") 