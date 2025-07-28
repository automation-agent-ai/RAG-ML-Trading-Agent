#!/usr/bin/env python3
"""
Train ML Models

This script trains various machine learning models using the merged features.

Usage:
    python train_ml_models.py --training-data data/merged_ml_features_training.csv --test-data data/merged_ml_features_prediction.csv --output-dir models
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLTrainer:
    """Class for training ML models"""
    
    def __init__(
        self,
        output_dir: str = "models",
        random_state: int = 42
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.random_state = random_state
        self.models = {}
        self.feature_importances = {}
        self.evaluation_results = {}
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        label_col: str = "label",
        exclude_cols: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training by separating features and labels
        
        Args:
            data: DataFrame with features and labels
            label_col: Name of the label column
            exclude_cols: Columns to exclude from features
            
        Returns:
            Tuple of (X, y) where X is features and y is labels
        """
        if exclude_cols is None:
            exclude_cols = []
        
        if label_col not in data.columns:
            raise ValueError(f"Label column '{label_col}' not found in data")
        
        # Get labels
        y = data[label_col]
        
        # Get features
        exclude_cols = exclude_cols + [label_col]
        X = data.drop(columns=exclude_cols, errors='ignore')
        
        logger.info(f"Prepared data with {X.shape[1]} features and {len(y)} samples")
        
        return X, y
    
    def train_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models_to_train: List[str] = None
    ) -> Dict[str, Any]:
        """
        Train multiple ML models
        
        Args:
            X: Feature DataFrame
            y: Label Series
            models_to_train: List of model names to train
            
        Returns:
            Dictionary of trained models
        """
        if models_to_train is None:
            models_to_train = ["random_forest", "gradient_boosting", "logistic_regression"]
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Train models
        for model_name in models_to_train:
            logger.info(f"Training {model_name}...")
            
            if model_name == "random_forest":
                model = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('classifier', RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=self.random_state
                    ))
                ])
            elif model_name == "gradient_boosting":
                model = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('classifier', GradientBoostingClassifier(
                        n_estimators=100,
                        max_depth=5,
                        random_state=self.random_state
                    ))
                ])
            elif model_name == "logistic_regression":
                model = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(
                        max_iter=1000,
                        random_state=self.random_state
                    ))
                ])
            else:
                logger.warning(f"Unknown model: {model_name}")
                continue
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Save the model
            self.models[model_name] = model
            
            # Evaluate on validation set
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='weighted')
            recall = recall_score(y_val, y_pred, average='weighted')
            f1 = f1_score(y_val, y_pred, average='weighted')
            cm = confusion_matrix(y_val, y_pred)
            
            # Save evaluation results
            self.evaluation_results[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm
            }
            
            # Extract feature importances if available
            if hasattr(model[-1], 'feature_importances_'):
                feature_importances = model[-1].feature_importances_
                feature_importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': feature_importances
                }).sort_values('importance', ascending=False)
                
                self.feature_importances[model_name] = feature_importance_df
            
            logger.info(f"  - Accuracy: {accuracy:.4f}")
            logger.info(f"  - Precision: {precision:.4f}")
            logger.info(f"  - Recall: {recall:.4f}")
            logger.info(f"  - F1 Score: {f1:.4f}")
        
        return self.models
    
    def save_models(self) -> Dict[str, str]:
        """
        Save trained models to disk
        
        Returns:
            Dictionary of model paths
        """
        model_paths = {}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            # Create model directory
            model_dir = self.output_dir / model_name
            model_dir.mkdir(exist_ok=True, parents=True)
            
            # Save model
            model_path = model_dir / f"{model_name}_{timestamp}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            model_paths[model_name] = str(model_path)
            
            logger.info(f"Saved {model_name} to {model_path}")
            
            # Save feature importances if available
            if model_name in self.feature_importances:
                feature_importance_path = model_dir / f"{model_name}_feature_importances_{timestamp}.csv"
                self.feature_importances[model_name].to_csv(feature_importance_path, index=False)
                logger.info(f"Saved {model_name} feature importances to {feature_importance_path}")
            
            # Save evaluation results
            if model_name in self.evaluation_results:
                eval_path = model_dir / f"{model_name}_evaluation_{timestamp}.txt"
                with open(eval_path, 'w') as f:
                    f.write(f"Accuracy: {self.evaluation_results[model_name]['accuracy']:.4f}\n")
                    f.write(f"Precision: {self.evaluation_results[model_name]['precision']:.4f}\n")
                    f.write(f"Recall: {self.evaluation_results[model_name]['recall']:.4f}\n")
                    f.write(f"F1 Score: {self.evaluation_results[model_name]['f1']:.4f}\n")
                    f.write("\nConfusion Matrix:\n")
                    f.write(str(self.evaluation_results[model_name]['confusion_matrix']))
                
                logger.info(f"Saved {model_name} evaluation results to {eval_path}")
        
        return model_paths
    
    def evaluate_on_test_data(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate trained models on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation results
        """
        test_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name} on test data...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)
            
            # Save test results
            test_results[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm
            }
            
            logger.info(f"  - Test Accuracy: {accuracy:.4f}")
            logger.info(f"  - Test Precision: {precision:.4f}")
            logger.info(f"  - Test Recall: {recall:.4f}")
            logger.info(f"  - Test F1 Score: {f1:.4f}")
        
        return test_results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train ML models')
    parser.add_argument('--training-data', required=True,
                       help='Path to training data CSV')
    parser.add_argument('--test-data',
                       help='Path to test data CSV')
    parser.add_argument('--output-dir', default='models',
                       help='Directory to save trained models')
    parser.add_argument('--label-col', default='label',
                       help='Name of the label column')
    parser.add_argument('--exclude-cols', nargs='+',
                       help='Columns to exclude from features')
    parser.add_argument('--models', nargs='+',
                       default=['random_forest', 'gradient_boosting', 'logistic_regression'],
                       help='Models to train')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Load training data
    logger.info(f"Loading training data from {args.training_data}")
    training_data = pd.read_csv(args.training_data)
    
    # Initialize trainer
    trainer = MLTrainer(
        output_dir=args.output_dir,
        random_state=args.random_state
    )
    
    # Prepare training data
    X, y = trainer.prepare_data(
        data=training_data,
        label_col=args.label_col,
        exclude_cols=args.exclude_cols
    )
    
    # Train models
    trainer.train_models(
        X=X,
        y=y,
        models_to_train=args.models
    )
    
    # Save models
    trainer.save_models()
    
    # Evaluate on test data if provided
    if args.test_data:
        logger.info(f"Loading test data from {args.test_data}")
        test_data = pd.read_csv(args.test_data)
        
        # Prepare test data
        X_test, y_test = trainer.prepare_data(
            data=test_data,
            label_col=args.label_col,
            exclude_cols=args.exclude_cols
        )
        
        # Evaluate on test data
        trainer.evaluate_on_test_data(
            X_test=X_test,
            y_test=y_test
        )

if __name__ == '__main__':
    main() 