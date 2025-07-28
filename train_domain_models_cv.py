#!/usr/bin/env python3
"""
Train Domain-Specific ML Models with Cross-Validation

This script trains separate machine learning models for text and financial features
using cross-validation to get more robust performance estimates.

Usage:
    python train_domain_models_cv.py --text-data data/ml_features/text_ml_features_training_labeled.csv 
                                   --financial-data data/ml_features/financial_ml_features_training_labeled.csv 
                                   --output-dir models
                                   --cv 5
                                   --scoring precision_weighted
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

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_validate, StratifiedKFold

# Try to import XGBoost, use GradientBoosting as fallback
try:
    from xgboost import XGBClassifier
    has_xgboost = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    has_xgboost = False
    print("XGBoost not found, using GradientBoostingClassifier as fallback")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DomainMLTrainerCV:
    """Class for training domain-specific ML models with cross-validation"""
    
    def __init__(
        self,
        domain: str,
        output_dir: str = "models",
        cv: int = 5,
        scoring: str = "precision_weighted",
        random_state: int = 42
    ):
        """
        Initialize the trainer
        
        Args:
            domain: Domain name ('text' or 'financial')
            output_dir: Directory to save models
            cv: Number of cross-validation folds
            scoring: Scoring metric for cross-validation
            random_state: Random state for reproducibility
        """
        self.domain = domain
        self.output_dir = Path(output_dir) / domain
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.models = {}
        self.cv_results = {}
        self.feature_importances = {}
        self.evaluation_results = {}
        self.feature_columns = []
    
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
        
        # Drop rows with NaN labels
        data = data.dropna(subset=[label_col])
        
        if len(data) == 0:
            raise ValueError(f"No valid data with non-NaN '{label_col}' values")
        
        # Get labels
        y = data[label_col]
        
        # Get features
        exclude_cols = exclude_cols + [label_col, "setup_id"]
        X = data.drop(columns=exclude_cols, errors='ignore')
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        logger.info(f"Prepared {self.domain} data with {X.shape[1]} features and {len(y)} samples")
        logger.info(f"Label distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_models_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models_to_train: List[str] = None
    ) -> Dict[str, Any]:
        """
        Train multiple ML models with cross-validation
        
        Args:
            X: Feature DataFrame
            y: Label Series
            models_to_train: List of model names to train
            
        Returns:
            Dictionary of trained models
        """
        if models_to_train is None:
            models_to_train = ["random_forest", "xgboost", "logistic_regression"]
        
        # Define cross-validation strategy
        cv_strategy = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        # Define scoring metrics
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        # Train models
        for model_name in models_to_train:
            logger.info(f"Training {self.domain} {model_name} with {self.cv}-fold cross-validation...")
            
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
            elif model_name == "xgboost":
                if has_xgboost:
                    model = Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                        ('classifier', XGBClassifier(
                            n_estimators=100,
                            max_depth=5,
                            random_state=self.random_state
                        ))
                    ])
                else:
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
            
            # Perform cross-validation
            cv_results = cross_validate(
                model, X, y, 
                cv=cv_strategy,
                scoring=scoring,
                return_estimator=True,
                n_jobs=-1
            )
            
            # Store CV results
            self.cv_results[model_name] = cv_results
            
            # Get best model based on precision_weighted
            best_idx = np.argmax(cv_results['test_precision_weighted'])
            best_model = cv_results['estimator'][best_idx]
            
            # Save the best model
            self.models[model_name] = best_model
            
            # Log CV results
            logger.info(f"  - Mean Accuracy: {np.mean(cv_results['test_accuracy']):.4f} (±{np.std(cv_results['test_accuracy']):.4f})")
            logger.info(f"  - Mean Precision: {np.mean(cv_results['test_precision_weighted']):.4f} (±{np.std(cv_results['test_precision_weighted']):.4f})")
            logger.info(f"  - Mean Recall: {np.mean(cv_results['test_recall_weighted']):.4f} (±{np.std(cv_results['test_recall_weighted']):.4f})")
            logger.info(f"  - Mean F1 Score: {np.mean(cv_results['test_f1_weighted']):.4f} (±{np.std(cv_results['test_f1_weighted']):.4f})")
            
            # Extract feature importances from best model if available
            if hasattr(best_model[-1], 'feature_importances_'):
                feature_importances = best_model[-1].feature_importances_
                feature_importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': feature_importances
                }).sort_values('importance', ascending=False)
                
                self.feature_importances[model_name] = feature_importance_df
                
                # Plot feature importances
                self._plot_feature_importances(model_name, feature_importance_df)
        
        return self.models
    
    def _plot_feature_importances(self, model_name: str, importance_df: pd.DataFrame):
        """Plot feature importances"""
        # Get top 20 features
        top_features = importance_df.head(20)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top 20 Feature Importances - {self.domain} {model_name}')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{model_name}_feature_importances.png"
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Saved feature importance plot to {plot_path}")
    
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
            model_path = model_dir / f"{model_name}_cv_{timestamp}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            model_paths[model_name] = str(model_path)
            
            logger.info(f"Saved {self.domain} {model_name} to {model_path}")
            
            # Save feature importances if available
            if model_name in self.feature_importances:
                feature_importance_path = model_dir / f"{model_name}_feature_importances_{timestamp}.csv"
                self.feature_importances[model_name].to_csv(feature_importance_path, index=False)
                logger.info(f"Saved {self.domain} {model_name} feature importances to {feature_importance_path}")
            
            # Save CV results
            if model_name in self.cv_results:
                cv_results_path = model_dir / f"{model_name}_cv_results_{timestamp}.txt"
                with open(cv_results_path, 'w') as f:
                    f.write(f"Domain: {self.domain}\n")
                    f.write(f"Model: {model_name}\n\n")
                    f.write(f"Cross-Validation Results ({self.cv} folds):\n")
                    f.write(f"  - Mean Accuracy: {np.mean(self.cv_results[model_name]['test_accuracy']):.4f} (±{np.std(self.cv_results[model_name]['test_accuracy']):.4f})\n")
                    f.write(f"  - Mean Precision: {np.mean(self.cv_results[model_name]['test_precision_weighted']):.4f} (±{np.std(self.cv_results[model_name]['test_precision_weighted']):.4f})\n")
                    f.write(f"  - Mean Recall: {np.mean(self.cv_results[model_name]['test_recall_weighted']):.4f} (±{np.std(self.cv_results[model_name]['test_recall_weighted']):.4f})\n")
                    f.write(f"  - Mean F1 Score: {np.mean(self.cv_results[model_name]['test_f1_weighted']):.4f} (±{np.std(self.cv_results[model_name]['test_f1_weighted']):.4f})\n")
                
                logger.info(f"Saved {self.domain} {model_name} CV results to {cv_results_path}")
        
        # Save model weights based on precision for ensemble
        weights_path = self.output_dir / f"model_weights_{timestamp}.csv"
        weights_df = pd.DataFrame({
            'model': list(self.cv_results.keys()),
            'precision': [np.mean(self.cv_results[model]['test_precision_weighted']) for model in self.cv_results],
            'weight': [np.mean(self.cv_results[model]['test_precision_weighted']) for model in self.cv_results]
        })
        weights_df.to_csv(weights_path, index=False)
        logger.info(f"Saved model weights to {weights_path}")
        
        return model_paths
    
    def evaluate_on_test_data(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        output_predictions: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate trained models on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            output_predictions: Whether to output predictions to CSV
            
        Returns:
            Dictionary of evaluation results
        """
        test_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {self.domain} {model_name} on test data...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            # Save test results
            test_results[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm,
                "classification_report": report,
                "predictions": y_pred
            }
            
            # Plot confusion matrix for test data
            self._plot_confusion_matrix(model_name, cm, y_test.unique())
            
            logger.info(f"  - Test Accuracy: {accuracy:.4f}")
            logger.info(f"  - Test Precision: {precision:.4f}")
            logger.info(f"  - Test Recall: {recall:.4f}")
            logger.info(f"  - Test F1 Score: {f1:.4f}")
            
            # Output predictions to CSV if requested
            if output_predictions:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                predictions_path = self.output_dir / f"{model_name}_predictions_{timestamp}.csv"
                
                # Create DataFrame with predictions
                predictions_df = pd.DataFrame({
                    'true_label': y_test,
                    'predicted_label': y_pred
                })
                
                # Save to CSV
                predictions_df.to_csv(predictions_path, index=False)
                logger.info(f"Saved {self.domain} {model_name} predictions to {predictions_path}")
        
        return test_results
    
    def _plot_confusion_matrix(self, model_name: str, cm: np.ndarray, classes: np.ndarray):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {self.domain} {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Saved confusion matrix plot to {plot_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train domain-specific ML models with cross-validation')
    parser.add_argument('--text-data', required=True,
                      help='Path to text ML features CSV')
    parser.add_argument('--financial-data', required=True,
                      help='Path to financial ML features CSV')
    parser.add_argument('--text-test-data',
                      help='Path to text ML features test CSV')
    parser.add_argument('--financial-test-data',
                      help='Path to financial ML features test CSV')
    parser.add_argument('--output-dir', default='models',
                      help='Directory to save trained models')
    parser.add_argument('--label-col', default='label',
                      help='Name of the label column')
    parser.add_argument('--exclude-cols', nargs='+',
                      default=['outperformance_10d'],
                      help='Columns to exclude from features')
    parser.add_argument('--models', nargs='+',
                      default=['random_forest', 'xgboost', 'logistic_regression'],
                      help='Models to train')
    parser.add_argument('--cv', type=int, default=5,
                      help='Number of cross-validation folds')
    parser.add_argument('--scoring', default='precision_weighted',
                      help='Scoring metric for cross-validation')
    parser.add_argument('--random-state', type=int, default=42,
                      help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Train text models
    logger.info("=== Training Text Models with Cross-Validation ===")
    
    # Load text training data
    logger.info(f"Loading text training data from {args.text_data}")
    text_data = pd.read_csv(args.text_data)
    
    # Initialize text trainer
    text_trainer = DomainMLTrainerCV(
        domain='text',
        output_dir=args.output_dir,
        cv=args.cv,
        scoring=args.scoring,
        random_state=args.random_state
    )
    
    # Prepare text training data
    X_text, y_text = text_trainer.prepare_data(
        data=text_data,
        label_col=args.label_col,
        exclude_cols=args.exclude_cols
    )
    
    # Train text models with cross-validation
    text_trainer.train_models_with_cv(
        X=X_text,
        y=y_text,
        models_to_train=args.models
    )
    
    # Save text models
    text_trainer.save_models()
    
    # Evaluate text models on test data if provided
    if args.text_test_data:
        logger.info(f"Loading text test data from {args.text_test_data}")
        text_test_data = pd.read_csv(args.text_test_data)
        
        # Prepare text test data
        X_text_test, y_text_test = text_trainer.prepare_data(
            data=text_test_data,
            label_col=args.label_col,
            exclude_cols=args.exclude_cols
        )
        
        # Evaluate text models on test data
        text_trainer.evaluate_on_test_data(
            X_test=X_text_test,
            y_test=y_text_test
        )
    
    # Train financial models
    logger.info("\n=== Training Financial Models with Cross-Validation ===")
    
    # Load financial training data
    logger.info(f"Loading financial training data from {args.financial_data}")
    financial_data = pd.read_csv(args.financial_data)
    
    # Initialize financial trainer
    financial_trainer = DomainMLTrainerCV(
        domain='financial',
        output_dir=args.output_dir,
        cv=args.cv,
        scoring=args.scoring,
        random_state=args.random_state
    )
    
    # Prepare financial training data
    X_financial, y_financial = financial_trainer.prepare_data(
        data=financial_data,
        label_col=args.label_col,
        exclude_cols=args.exclude_cols
    )
    
    # Train financial models with cross-validation
    financial_trainer.train_models_with_cv(
        X=X_financial,
        y=y_financial,
        models_to_train=args.models
    )
    
    # Save financial models
    financial_trainer.save_models()
    
    # Evaluate financial models on test data if provided
    if args.financial_test_data:
        logger.info(f"Loading financial test data from {args.financial_test_data}")
        financial_test_data = pd.read_csv(args.financial_test_data)
        
        # Prepare financial test data
        X_financial_test, y_financial_test = financial_trainer.prepare_data(
            data=financial_test_data,
            label_col=args.label_col,
            exclude_cols=args.exclude_cols
        )
        
        # Evaluate financial models on test data
        financial_trainer.evaluate_on_test_data(
            X_test=X_financial_test,
            y_test=y_financial_test
        )
    
    logger.info("\n=== Training Complete ===")
    logger.info(f"Models saved to {args.output_dir}")

if __name__ == '__main__':
    main() 