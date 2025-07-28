#!/usr/bin/env python3
"""
Ensemble Domain Predictions

This script combines predictions from text and financial models to create ensemble predictions.

Usage:
    python ensemble_domain_predictions.py --text-models models/text --financial-models models/financial --test-data data/ml_features/text_ml_features_prediction_*.csv --financial-test-data data/ml_features/financial_ml_features_prediction_*.csv --output ensemble_predictions.csv
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
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

class EnsemblePredictor:
    """Class for ensembling predictions from different domains"""
    
    def __init__(
        self,
        output_dir: str = "ensemble",
        ensemble_method: str = "voting"
    ):
        """
        Initialize the ensemble predictor
        
        Args:
            output_dir: Directory to save ensemble results
            ensemble_method: Method for ensembling ('voting' or 'stacking')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.ensemble_method = ensemble_method
        self.domain_models = {}
        self.domain_predictions = {}
        self.ensemble_predictions = None
    
    def load_models(self, domain: str, model_dir: str, model_types: List[str] = None) -> Dict[str, Any]:
        """
        Load trained models for a domain
        
        Args:
            domain: Domain name ('text' or 'financial')
            model_dir: Directory containing trained models
            model_types: Types of models to load ('random_forest', 'xgboost', 'logistic_regression')
            
        Returns:
            Dictionary of loaded models
        """
        if model_types is None:
            model_types = ['random_forest', 'xgboost', 'logistic_regression']
        
        domain_models = {}
        
        # Find model files
        model_dir_path = Path(model_dir)
        for model_type in model_types:
            model_type_dir = model_dir_path / model_type
            if not model_type_dir.exists():
                logger.warning(f"Model directory {model_type_dir} not found")
                continue
            
            # Find the latest model file
            model_files = list(model_type_dir.glob(f"{model_type}_*.pkl"))
            if not model_files:
                logger.warning(f"No model files found in {model_type_dir}")
                continue
            
            # Sort by modification time (newest first)
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_model_file = model_files[0]
            
            # Load the model
            with open(latest_model_file, 'rb') as f:
                model = pickle.load(f)
            
            domain_models[model_type] = model
            logger.info(f"Loaded {domain} {model_type} model from {latest_model_file}")
        
        if not domain_models:
            logger.warning(f"No models loaded for {domain}")
        
        self.domain_models[domain] = domain_models
        return domain_models
    
    def predict(
        self,
        domain: str,
        X: pd.DataFrame,
        setup_ids: pd.Series = None
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions for a domain
        
        Args:
            domain: Domain name ('text' or 'financial')
            X: Feature DataFrame
            setup_ids: Setup IDs (optional)
            
        Returns:
            Dictionary of predictions for each model
        """
        if domain not in self.domain_models:
            raise ValueError(f"No models loaded for {domain}")
        
        domain_predictions = {}
        
        for model_type, model in self.domain_models[domain].items():
            # Make predictions
            y_pred = model.predict(X)
            y_pred_proba = None
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X)
                except:
                    pass
            
            domain_predictions[model_type] = {
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"Made predictions with {domain} {model_type} model")
        
        self.domain_predictions[domain] = domain_predictions
        return domain_predictions
    
    def ensemble_predictions(
        self,
        setup_ids: pd.Series,
        method: str = None,
        weights: Dict[str, Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Ensemble predictions from different domains
        
        Args:
            setup_ids: Setup IDs
            method: Ensemble method ('voting' or 'stacking')
            weights: Dictionary of weights for each domain and model
            
        Returns:
            DataFrame with ensemble predictions
        """
        if method is None:
            method = self.ensemble_method
        
        if not self.domain_predictions:
            raise ValueError("No predictions to ensemble")
        
        # Default weights (equal weighting)
        if weights is None:
            weights = {}
            for domain in self.domain_predictions:
                weights[domain] = {}
                for model_type in self.domain_predictions[domain]:
                    weights[domain][model_type] = 1.0
        
        # Create a DataFrame to store all predictions
        all_predictions = pd.DataFrame({'setup_id': setup_ids})
        
        # Add individual model predictions to the DataFrame
        for domain in self.domain_predictions:
            for model_type, preds in self.domain_predictions[domain].items():
                col_name = f"{domain}_{model_type}"
                all_predictions[col_name] = preds['predictions']
        
        # Perform ensembling
        if method == 'voting':
            # Majority voting
            # Count votes for each class
            class_counts = {}
            for domain in self.domain_predictions:
                for model_type, preds in self.domain_predictions[domain].items():
                    weight = weights.get(domain, {}).get(model_type, 1.0)
                    for i, pred in enumerate(preds['predictions']):
                        if i not in class_counts:
                            class_counts[i] = {}
                        if pred not in class_counts[i]:
                            class_counts[i][pred] = 0
                        class_counts[i][pred] += weight
            
            # Get majority class for each sample
            ensemble_preds = []
            for i in range(len(setup_ids)):
                if i in class_counts:
                    # Get class with highest vote count
                    majority_class = max(class_counts[i].items(), key=lambda x: x[1])[0]
                    ensemble_preds.append(majority_class)
                else:
                    ensemble_preds.append(None)
            
            all_predictions['ensemble_prediction'] = ensemble_preds
        
        elif method == 'stacking':
            # Simple averaging of predictions (weighted)
            # This is a simplified version of stacking
            # For true stacking, we would need to train a meta-model
            
            # Get unique classes
            all_classes = set()
            for domain in self.domain_predictions:
                for model_type, preds in self.domain_predictions[domain].items():
                    all_classes.update(np.unique(preds['predictions']))
            
            # Convert to list and sort
            all_classes = sorted(list(all_classes))
            
            # Initialize class probabilities
            class_probs = {cls: np.zeros(len(setup_ids)) for cls in all_classes}
            total_weights = np.zeros(len(setup_ids))
            
            # Sum weighted probabilities
            for domain in self.domain_predictions:
                for model_type, preds in self.domain_predictions[domain].items():
                    weight = weights.get(domain, {}).get(model_type, 1.0)
                    for i, pred in enumerate(preds['predictions']):
                        class_probs[pred][i] += weight
                        total_weights[i] += weight
            
            # Normalize probabilities
            for cls in all_classes:
                class_probs[cls] = np.divide(class_probs[cls], total_weights, where=total_weights!=0)
            
            # Get class with highest probability for each sample
            ensemble_preds = []
            for i in range(len(setup_ids)):
                max_prob = -1
                max_class = None
                for cls in all_classes:
                    if class_probs[cls][i] > max_prob:
                        max_prob = class_probs[cls][i]
                        max_class = cls
                ensemble_preds.append(max_class)
            
            all_predictions['ensemble_prediction'] = ensemble_preds
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        self.ensemble_predictions = all_predictions
        return all_predictions
    
    def evaluate(
        self,
        y_true: pd.Series,
        output_report: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate predictions against true labels
        
        Args:
            y_true: True labels
            output_report: Whether to output evaluation report
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.ensemble_predictions is None:
            raise ValueError("No ensemble predictions to evaluate")
        
        # Evaluate individual model predictions
        model_metrics = {}
        for col in self.ensemble_predictions.columns:
            if col == 'setup_id':
                continue
            
            y_pred = self.ensemble_predictions[col]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_true, y_pred),
                'classification_report': classification_report(y_true, y_pred)
            }
            
            model_metrics[col] = metrics
            
            # Plot confusion matrix
            self._plot_confusion_matrix(col, metrics['confusion_matrix'], np.unique(y_true))
            
            logger.info(f"Metrics for {col}:")
            logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  - Precision: {metrics['precision']:.4f}")
            logger.info(f"  - Recall: {metrics['recall']:.4f}")
            logger.info(f"  - F1 Score: {metrics['f1']:.4f}")
        
        # Output evaluation report
        if output_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"ensemble_evaluation_{timestamp}.txt"
            
            with open(report_path, 'w') as f:
                f.write("Ensemble Evaluation Report\n")
                f.write("========================\n\n")
                
                for model_name, metrics in model_metrics.items():
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                    f.write(f"Precision: {metrics['precision']:.4f}\n")
                    f.write(f"Recall: {metrics['recall']:.4f}\n")
                    f.write(f"F1 Score: {metrics['f1']:.4f}\n\n")
                    f.write("Confusion Matrix:\n")
                    f.write(str(metrics['confusion_matrix']))
                    f.write("\n\nClassification Report:\n")
                    f.write(str(metrics['classification_report']))
                    f.write("\n\n")
            
            logger.info(f"Saved evaluation report to {report_path}")
        
        return model_metrics
    
    def _plot_confusion_matrix(self, model_name: str, cm: np.ndarray, classes: np.ndarray):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Saved confusion matrix plot to {plot_path}")
    
    def save_predictions(self, output_file: str = None) -> str:
        """
        Save ensemble predictions to CSV
        
        Args:
            output_file: Path to output CSV file
            
        Returns:
            Path to saved CSV file
        """
        if self.ensemble_predictions is None:
            raise ValueError("No ensemble predictions to save")
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"ensemble_predictions_{timestamp}.csv"
        else:
            output_file = Path(output_file)
        
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        # Save to CSV
        self.ensemble_predictions.to_csv(output_file, index=False)
        logger.info(f"Saved ensemble predictions to {output_file}")
        
        return str(output_file)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Ensemble domain predictions')
    parser.add_argument('--text-models', required=True,
                       help='Directory containing trained text models')
    parser.add_argument('--financial-models', required=True,
                       help='Directory containing trained financial models')
    parser.add_argument('--text-test-data', required=True,
                       help='Path to text test data CSV')
    parser.add_argument('--financial-test-data', required=True,
                       help='Path to financial test data CSV')
    parser.add_argument('--output', default='data/ensemble_predictions.csv',
                       help='Path to output CSV file')
    parser.add_argument('--output-dir', default='ensemble',
                       help='Directory to save ensemble results')
    parser.add_argument('--ensemble-method', choices=['voting', 'stacking'], default='voting',
                       help='Method for ensembling predictions')
    parser.add_argument('--label-col', default='label',
                       help='Name of the label column')
    parser.add_argument('--setup-id-col', default='setup_id',
                       help='Name of the setup ID column')
    parser.add_argument('--model-types', nargs='+',
                       default=['random_forest', 'xgboost', 'logistic_regression'],
                       help='Types of models to load')
    
    args = parser.parse_args()
    
    # Initialize ensemble predictor
    ensemble_predictor = EnsemblePredictor(
        output_dir=args.output_dir,
        ensemble_method=args.ensemble_method
    )
    
    # Load text models
    ensemble_predictor.load_models(
        domain='text',
        model_dir=args.text_models,
        model_types=args.model_types
    )
    
    # Load financial models
    ensemble_predictor.load_models(
        domain='financial',
        model_dir=args.financial_models,
        model_types=args.model_types
    )
    
    # Load text test data
    logger.info(f"Loading text test data from {args.text_test_data}")
    text_test_data = pd.read_csv(args.text_test_data)
    
    # Load financial test data
    logger.info(f"Loading financial test data from {args.financial_test_data}")
    financial_test_data = pd.read_csv(args.financial_test_data)
    
    # Get setup IDs
    setup_ids = text_test_data[args.setup_id_col]
    
    # Check if setup IDs match
    if not setup_ids.equals(financial_test_data[args.setup_id_col]):
        logger.warning("Setup IDs in text and financial test data do not match")
        # Use intersection of setup IDs
        common_setup_ids = set(setup_ids) & set(financial_test_data[args.setup_id_col])
        logger.info(f"Using {len(common_setup_ids)} common setup IDs")
        
        # Filter data to common setup IDs
        text_test_data = text_test_data[text_test_data[args.setup_id_col].isin(common_setup_ids)]
        financial_test_data = financial_test_data[financial_test_data[args.setup_id_col].isin(common_setup_ids)]
        
        # Update setup IDs
        setup_ids = text_test_data[args.setup_id_col]
    
    # Prepare text features
    X_text = text_test_data.drop(columns=[args.setup_id_col, args.label_col], errors='ignore')
    
    # Prepare financial features
    X_financial = financial_test_data.drop(columns=[args.setup_id_col, args.label_col], errors='ignore')
    
    # Make predictions with text models
    ensemble_predictor.predict(
        domain='text',
        X=X_text,
        setup_ids=setup_ids
    )
    
    # Make predictions with financial models
    ensemble_predictor.predict(
        domain='financial',
        X=X_financial,
        setup_ids=setup_ids
    )
    
    # Ensemble predictions
    ensemble_predictions = ensemble_predictor.ensemble_predictions(
        setup_ids=setup_ids,
        method=args.ensemble_method
    )
    
    # Save predictions
    ensemble_predictor.save_predictions(args.output)
    
    # Evaluate if label column is available
    if args.label_col in text_test_data.columns:
        y_true = text_test_data[args.label_col]
        ensemble_predictor.evaluate(y_true=y_true)
    
    logger.info("Ensemble prediction complete")

if __name__ == '__main__':
    main() 