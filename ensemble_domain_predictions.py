#!/usr/bin/env python3
"""
Ensemble Domain-Specific Model Predictions

This script combines predictions from text and financial domain models
to make ensemble predictions.

Usage:
    python ensemble_domain_predictions.py --text-data data/ml_features/text_ml_features_prediction_labeled.csv 
                                         --financial-data data/ml_features/financial_ml_features_prediction_labeled.csv 
                                         --text-models-dir models/text 
                                         --financial-models-dir models/financial 
                                         --output-file data/ensemble_predictions.csv
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import pickle
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DomainEnsembler:
    """Class for ensembling domain-specific model predictions"""
    
    def __init__(
        self,
        text_models_dir: str = "models/text",
        financial_models_dir: str = "models/financial",
        ensemble_method: str = "weighted_voting"
    ):
        """
        Initialize the ensembler
        
        Args:
            text_models_dir: Directory containing text models
            financial_models_dir: Directory containing financial models
            ensemble_method: Method for ensembling predictions
                - 'majority_voting': Simple majority voting
                - 'weighted_voting': Weighted voting based on model performance
        """
        self.text_models_dir = Path(text_models_dir)
        self.financial_models_dir = Path(financial_models_dir)
        self.ensemble_method = ensemble_method
        self.text_models = {}
        self.financial_models = {}
        self.text_weights = {}
        self.financial_weights = {}
    
    def load_latest_models(self):
        """
        Load the latest trained models from each domain
        """
        # Load text models
        self._load_domain_models(self.text_models_dir, self.text_models)
        
        # Load financial models
        self._load_domain_models(self.financial_models_dir, self.financial_models)
        
        # Log loaded models
        logger.info(f"Loaded {len(self.text_models)} text models and {len(self.financial_models)} financial models")
    
    def _load_domain_models(self, models_dir: Path, models_dict: Dict):
        """Helper to load models from a domain directory"""
        # Get model subdirectories
        model_dirs = [d for d in models_dir.glob("*") if d.is_dir()]
        
        for model_dir in model_dirs:
            model_name = model_dir.name
            
            # Find the latest model file
            model_files = list(model_dir.glob("*.pkl"))
            if not model_files:
                logger.warning(f"No model files found in {model_dir}")
                continue
            
            # Sort by modification time (newest first)
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_model_file = model_files[0]
            
            # Load the model
            try:
                with open(latest_model_file, 'rb') as f:
                    model = pickle.load(f)
                models_dict[model_name] = model
                logger.info(f"Loaded {model_name} model from {latest_model_file}")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
    
    def set_model_weights(self, text_weights: Dict[str, float] = None, financial_weights: Dict[str, float] = None):
        """
        Set weights for models in each domain
        
        Args:
            text_weights: Dictionary mapping text model names to weights
            financial_weights: Dictionary mapping financial model names to weights
        """
        if text_weights:
            self.text_weights = text_weights
        else:
            # Equal weights by default
            self.text_weights = {name: 1.0 for name in self.text_models}
        
        if financial_weights:
            self.financial_weights = financial_weights
        else:
            # Equal weights by default
            self.financial_weights = {name: 1.0 for name in self.financial_models}
    
    def predict(
        self,
        text_data: pd.DataFrame,
        financial_data: pd.DataFrame,
        exclude_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Make ensemble predictions using text and financial models
        
        Args:
            text_data: DataFrame with text features
            financial_data: DataFrame with financial features
            exclude_cols: Columns to exclude from features
            
        Returns:
            DataFrame with ensemble predictions
        """
        if exclude_cols is None:
            exclude_cols = []
        
        # Get setup IDs
        text_setup_ids = text_data['setup_id'].tolist()
        financial_setup_ids = financial_data['setup_id'].tolist()
        
        # Check that the setup IDs match
        if set(text_setup_ids) != set(financial_setup_ids):
            logger.warning("Text and financial data have different setup IDs")
            # Use intersection of setup IDs
            common_setup_ids = set(text_setup_ids).intersection(set(financial_setup_ids))
            text_data = text_data[text_data['setup_id'].isin(common_setup_ids)]
            financial_data = financial_data[financial_data['setup_id'].isin(common_setup_ids)]
            logger.info(f"Using {len(common_setup_ids)} common setup IDs")
        
        # Get true labels if available
        if 'label' in text_data.columns:
            true_labels = text_data['label'].tolist()
        else:
            true_labels = None
        
        # Prepare features
        exclude_cols = exclude_cols + ['setup_id', 'label']
        
        # Make sure outperformance_10d is present (models expect it)
        if 'outperformance_10d' in text_data.columns:
            text_features = text_data.drop(columns=exclude_cols + ['outperformance_10d'], errors='ignore')
        else:
            # Add dummy outperformance_10d column filled with zeros
            text_features = text_data.drop(columns=exclude_cols, errors='ignore').copy()
            text_features['outperformance_10d'] = 0.0
            
        if 'outperformance_10d' in financial_data.columns:
            financial_features = financial_data.drop(columns=exclude_cols + ['outperformance_10d'], errors='ignore')
        else:
            # Add dummy outperformance_10d column filled with zeros
            financial_features = financial_data.drop(columns=exclude_cols, errors='ignore').copy()
            financial_features['outperformance_10d'] = 0.0
        
        # Make predictions with each model
        text_predictions = {}
        financial_predictions = {}
        
        # Text model predictions
        for model_name, model in self.text_models.items():
            try:
                preds = model.predict(text_features)
                text_predictions[model_name] = preds
                logger.info(f"Made predictions with {model_name} text model")
            except Exception as e:
                logger.error(f"Error making predictions with {model_name} text model: {e}")
        
        # Financial model predictions
        for model_name, model in self.financial_models.items():
            try:
                preds = model.predict(financial_features)
                financial_predictions[model_name] = preds
                logger.info(f"Made predictions with {model_name} financial model")
            except Exception as e:
                logger.error(f"Error making predictions with {model_name} financial model: {e}")
        
        # Ensemble predictions
        if self.ensemble_method == 'majority_voting':
            ensemble_preds = self._majority_voting(text_predictions, financial_predictions)
        elif self.ensemble_method == 'weighted_voting':
            ensemble_preds = self._weighted_voting(text_predictions, financial_predictions)
        else:
            logger.error(f"Unknown ensemble method: {self.ensemble_method}")
            ensemble_preds = self._majority_voting(text_predictions, financial_predictions)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'setup_id': text_data['setup_id'].tolist(),
            'ensemble_prediction': ensemble_preds
        })
        
        # Add individual model predictions
        for model_name, preds in text_predictions.items():
            results[f'text_{model_name}_prediction'] = preds
        
        for model_name, preds in financial_predictions.items():
            results[f'financial_{model_name}_prediction'] = preds
        
        # Add true labels if available
        if true_labels is not None:
            results['true_label'] = true_labels
        
        # Evaluate ensemble predictions if true labels are available
        if true_labels is not None:
            accuracy = accuracy_score(true_labels, ensemble_preds)
            precision = precision_score(true_labels, ensemble_preds, average='weighted')
            recall = recall_score(true_labels, ensemble_preds, average='weighted')
            f1 = f1_score(true_labels, ensemble_preds, average='weighted')
            
            logger.info(f"Ensemble Accuracy: {accuracy:.4f}")
            logger.info(f"Ensemble Precision: {precision:.4f}")
            logger.info(f"Ensemble Recall: {recall:.4f}")
            logger.info(f"Ensemble F1 Score: {f1:.4f}")
            
            # Add evaluation metrics to results
            results_metrics = pd.DataFrame({
                'metric': ['accuracy', 'precision', 'recall', 'f1'],
                'value': [accuracy, precision, recall, f1]
            })
            
            # Save metrics to CSV
            metrics_file = Path('data') / f'ensemble_predictions_metrics.csv'
            results_metrics.to_csv(metrics_file, index=False)
            logger.info(f"Saved ensemble metrics to {metrics_file}")
        
        return results
    
    def _majority_voting(
        self,
        text_predictions: Dict[str, np.ndarray],
        financial_predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Combine predictions using majority voting
        
        Args:
            text_predictions: Dictionary mapping text model names to predictions
            financial_predictions: Dictionary mapping financial model names to predictions
            
        Returns:
            Array of ensemble predictions
        """
        # Combine all predictions
        all_preds = []
        for preds in text_predictions.values():
            all_preds.append(preds)
        
        for preds in financial_predictions.values():
            all_preds.append(preds)
        
        # Stack predictions and take mode along axis 0
        if all_preds:
            stacked_preds = np.vstack(all_preds)
            # Count occurrences of each class for each sample
            ensemble_preds = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int), minlength=3).argmax(),
                axis=0,
                arr=stacked_preds
            )
            return ensemble_preds
        else:
            logger.error("No predictions to ensemble")
            return np.array([])
    
    def _weighted_voting(
        self,
        text_predictions: Dict[str, np.ndarray],
        financial_predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Combine predictions using weighted voting
        
        Args:
            text_predictions: Dictionary mapping text model names to predictions
            financial_predictions: Dictionary mapping financial model names to predictions
            
        Returns:
            Array of ensemble predictions
        """
        # Get number of samples
        n_samples = 0
        for preds in text_predictions.values():
            n_samples = len(preds)
            break
        
        if n_samples == 0:
            for preds in financial_predictions.values():
                n_samples = len(preds)
                break
        
        if n_samples == 0:
            logger.error("No predictions to ensemble")
            return np.array([])
        
        # Initialize vote counts for each class
        votes = np.zeros((n_samples, 3))  # 3 classes: 0, 1, 2
        
        # Add weighted votes for text models
        for model_name, preds in text_predictions.items():
            weight = self.text_weights.get(model_name, 1.0)
            for i, pred in enumerate(preds):
                votes[i, int(pred)] += weight
        
        # Add weighted votes for financial models
        for model_name, preds in financial_predictions.items():
            weight = self.financial_weights.get(model_name, 1.0)
            for i, pred in enumerate(preds):
                votes[i, int(pred)] += weight
        
        # Get class with highest weighted vote for each sample
        ensemble_preds = np.argmax(votes, axis=1)
        return ensemble_preds

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Ensemble domain-specific model predictions')
    parser.add_argument('--text-data', required=True,
                       help='Path to text features CSV')
    parser.add_argument('--financial-data', required=True,
                       help='Path to financial features CSV')
    parser.add_argument('--text-models-dir', default='models/text',
                       help='Directory containing text models')
    parser.add_argument('--financial-models-dir', default='models/financial',
                       help='Directory containing financial models')
    parser.add_argument('--ensemble-method', choices=['majority_voting', 'weighted_voting'],
                       default='weighted_voting',
                       help='Method for ensembling predictions')
    parser.add_argument('--output-file', default='data/ensemble_predictions.csv',
                       help='Path to output CSV file')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading text data from {args.text_data}")
    text_data = pd.read_csv(args.text_data)
    
    logger.info(f"Loading financial data from {args.financial_data}")
    financial_data = pd.read_csv(args.financial_data)
    
    # Initialize ensembler
    ensembler = DomainEnsembler(
        text_models_dir=args.text_models_dir,
        financial_models_dir=args.financial_models_dir,
        ensemble_method=args.ensemble_method
    )
    
    # Load models
    ensembler.load_latest_models()
    
    # Set model weights (based on model performance)
    text_weights = {
        'random_forest': 1.0,
        'xgboost': 2.0,  # Increased weight for text XGBoost model
        'logistic_regression': 0.5
    }
    
    financial_weights = {
        'random_forest': 0.8,
        'xgboost': 0.8,
        'logistic_regression': 0.8
    }
    
    ensembler.set_model_weights(text_weights, financial_weights)
    
    # Make ensemble predictions
    results = ensembler.predict(text_data, financial_data)
    
    # Save results to CSV
    results.to_csv(args.output_file, index=False)
    logger.info(f"Saved ensemble predictions to {args.output_file}")

if __name__ == '__main__':
    main() 