#!/usr/bin/env python3
"""
Train Domain-Specific ML Models

This script trains separate machine learning models for text and financial features.
The models can then be ensembled to make final predictions.

Usage:
    python train_domain_models.py --text-data data/ml_features/text_ml_features_training_*.csv --financial-data data/ml_features/financial_ml_features_training_*.csv --output-dir models
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
from sklearn.model_selection import train_test_split

# Import financial preprocessor for consistent preprocessing
try:
    from financial_preprocessor import FinancialPreprocessor
    HAS_FINANCIAL_PREPROCESSOR = True
except ImportError:
    HAS_FINANCIAL_PREPROCESSOR = False
    logger = logging.getLogger(__name__)
    logger.warning("Financial preprocessor not found. Will use standard preprocessing for financial domain.")

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

class DomainMLTrainer:
    """Class for training domain-specific ML models"""
    
    def __init__(
        self,
        domain: str,
        output_dir: str = "models",
        random_state: int = 42
    ):
        """
        Initialize the trainer
        
        Args:
            domain: Domain name ('text' or 'financial')
            output_dir: Directory to save models
            random_state: Random state for reproducibility
        """
        self.domain = domain
        self.output_dir = Path(output_dir) / domain
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.random_state = random_state
        self.models = {}
        self.feature_importances = {}
        self.evaluation_results = {}
        self.feature_columns = []
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        label_col: str = "label",
        exclude_cols: List[str] = None,
        impute_strategy: str = "median",
        use_saved_preprocessing: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training by separating features and labels
        
        Args:
            data: DataFrame with features and labels
            label_col: Name of the label column
            exclude_cols: Columns to exclude from features
            impute_strategy: Strategy for imputing missing values
            use_saved_preprocessing: Whether to use saved preprocessing parameters for financial domain
            
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
        
        # For financial domain, try to use saved preprocessing parameters for consistency
        if (self.domain == 'financial' and HAS_FINANCIAL_PREPROCESSOR and 
            use_saved_preprocessing and hasattr(self, '_check_for_saved_preprocessing')):
            
            X = self._apply_financial_preprocessing(X, data)
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        logger.info(f"Prepared {self.domain} data with {X.shape[1]} features and {len(y)} samples")
        logger.info(f"Label distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def _check_for_saved_preprocessing(self) -> bool:
        """Check if saved preprocessing parameters exist for financial domain"""
        if self.domain != 'financial' or not HAS_FINANCIAL_PREPROCESSOR:
            return False
        
        preprocessing_path = Path(f"models/financial/preprocessing_params.pkl")
        return preprocessing_path.exists()
    
    def _apply_financial_preprocessing(self, X: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        """Apply saved financial preprocessing parameters for consistency"""
        try:
            logger.info(f"🔧 Loading saved financial preprocessing parameters for consistency...")
            
            # Initialize financial preprocessor
            preprocessor = FinancialPreprocessor(
                preprocessing_params_path="models/financial/preprocessing_params.pkl"
            )
            
            # Load preprocessing parameters
            preprocessor._load_preprocessing_params()
            
            # Check if feature columns match
            saved_features = set(preprocessor.feature_columns)
            current_features = set(X.columns)
            
            if saved_features != current_features:
                logger.warning(f"Feature mismatch detected:")
                logger.warning(f"  - Saved features: {len(saved_features)}")
                logger.warning(f"  - Current features: {len(current_features)}")
                
                # Find missing and extra features
                missing_features = saved_features - current_features
                extra_features = current_features - saved_features
                
                if missing_features:
                    logger.warning(f"  - Missing features: {list(missing_features)[:5]}...")
                    # Add missing features with NaN values
                    for feature in missing_features:
                        X[feature] = np.nan
                
                if extra_features:
                    logger.warning(f"  - Extra features: {list(extra_features)[:5]}...")
                    # Remove extra features
                    X = X.drop(columns=list(extra_features), errors='ignore')
                
                # Reorder columns to match saved order
                X = X[preprocessor.feature_columns]
            
            # Apply preprocessing (imputation and scaling)
            X_processed = preprocessor._apply_preprocessing(
                pd.concat([original_data[['setup_id', 'ticker']], X], axis=1),
                preprocessor.feature_columns
            )
            
            # Return only the feature columns
            X_final = X_processed[preprocessor.feature_columns]
            
            logger.info(f"✅ Applied saved financial preprocessing parameters")
            logger.info(f"  - Features: {X_final.shape[1]}")
            logger.info(f"  - Samples: {X_final.shape[0]}")
            
            return X_final
            
        except Exception as e:
            logger.warning(f"Failed to apply saved preprocessing: {str(e)}")
            logger.warning("Falling back to standard preprocessing pipeline")
            return X
    
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
            models_to_train = ["random_forest", "xgboost", "logistic_regression"]
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Check if we should use simplified preprocessing for financial domain
        use_simple_preprocessing = (
            self.domain == 'financial' and 
            HAS_FINANCIAL_PREPROCESSOR and 
            self._check_for_saved_preprocessing()
        )
        
        if use_simple_preprocessing:
            logger.info(f"🎯 Using simplified preprocessing for {self.domain} domain (preprocessing already applied)")
        
        # Train models
        for model_name in models_to_train:
            logger.info(f"Training {self.domain} {model_name}...")
            
            if model_name == "random_forest":
                if use_simple_preprocessing:
                    # Skip preprocessing since it's already applied
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=self.random_state
                    )
                else:
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
                    if use_simple_preprocessing:
                        model = XGBClassifier(
                            n_estimators=100,
                            max_depth=5,
                            random_state=self.random_state
                        )
                    else:
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
                    if use_simple_preprocessing:
                        model = GradientBoostingClassifier(
                            n_estimators=100,
                            max_depth=5,
                            random_state=self.random_state
                        )
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
                if use_simple_preprocessing:
                    model = LogisticRegression(
                        max_iter=1000,
                        random_state=self.random_state
                    )
                else:
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
            report = classification_report(y_val, y_pred)
            
            # Save evaluation results
            self.evaluation_results[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm,
                "classification_report": report
            }
            
            # Extract feature importances if available
            # Handle both Pipeline and non-Pipeline models
            classifier = model[-1] if hasattr(model, '__getitem__') else model
            
            if hasattr(classifier, 'feature_importances_'):
                feature_importances = classifier.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': feature_importances
                }).sort_values('importance', ascending=False)
                
                self.feature_importances[model_name] = feature_importance_df
                
                # Plot feature importances
                self._plot_feature_importances(model_name, feature_importance_df)
            
            # Plot confusion matrix
            self._plot_confusion_matrix(model_name, cm, y_val.unique())
            
            logger.info(f"  - Accuracy: {accuracy:.4f}")
            logger.info(f"  - Precision: {precision:.4f}")
            logger.info(f"  - Recall: {recall:.4f}")
            logger.info(f"  - F1 Score: {f1:.4f}")
        
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
            
            logger.info(f"Saved {self.domain} {model_name} to {model_path}")
            
            # Save feature importances if available
            if model_name in self.feature_importances:
                feature_importance_path = model_dir / f"{model_name}_feature_importances_{timestamp}.csv"
                self.feature_importances[model_name].to_csv(feature_importance_path, index=False)
                logger.info(f"Saved {self.domain} {model_name} feature importances to {feature_importance_path}")
            
            # Save evaluation results
            if model_name in self.evaluation_results:
                eval_path = model_dir / f"{model_name}_evaluation_{timestamp}.txt"
                with open(eval_path, 'w') as f:
                    f.write(f"Domain: {self.domain}\n")
                    f.write(f"Model: {model_name}\n\n")
                    f.write(f"Accuracy: {self.evaluation_results[model_name]['accuracy']:.4f}\n")
                    f.write(f"Precision: {self.evaluation_results[model_name]['precision']:.4f}\n")
                    f.write(f"Recall: {self.evaluation_results[model_name]['recall']:.4f}\n")
                    f.write(f"F1 Score: {self.evaluation_results[model_name]['f1']:.4f}\n\n")
                    f.write("Confusion Matrix:\n")
                    f.write(str(self.evaluation_results[model_name]['confusion_matrix']))
                    f.write("\n\nClassification Report:\n")
                    f.write(str(self.evaluation_results[model_name]['classification_report']))
                
                logger.info(f"Saved {self.domain} {model_name} evaluation results to {eval_path}")
        
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
            self._plot_confusion_matrix(f"{model_name}_test", cm, y_test.unique())
            
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

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train domain-specific ML models with enhanced financial preprocessing')
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
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--disable-saved-preprocessing', action='store_true',
                       help='Disable loading saved preprocessing parameters for financial domain')
    
    args = parser.parse_args()
    
    # Train text models
    logger.info("=== Training Text Models ===")
    
    # Load text training data
    logger.info(f"Loading text training data from {args.text_data}")
    text_data = pd.read_csv(args.text_data)
    
    # Initialize text trainer
    text_trainer = DomainMLTrainer(
        domain='text',
        output_dir=args.output_dir,
        random_state=args.random_state
    )
    
    # Prepare text training data
    X_text, y_text = text_trainer.prepare_data(
        data=text_data,
        label_col=args.label_col,
        exclude_cols=args.exclude_cols,
        use_saved_preprocessing=not args.disable_saved_preprocessing
    )
    
    # Train text models
    text_trainer.train_models(
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
            exclude_cols=args.exclude_cols,
            use_saved_preprocessing=not args.disable_saved_preprocessing
        )
        
        # Evaluate text models on test data
        text_trainer.evaluate_on_test_data(
            X_test=X_text_test,
            y_test=y_text_test
        )
    
    # Train financial models
    logger.info("\n=== Training Financial Models ===")
    
    # Load financial training data
    logger.info(f"Loading financial training data from {args.financial_data}")
    financial_data = pd.read_csv(args.financial_data)
    
    # Initialize financial trainer
    financial_trainer = DomainMLTrainer(
        domain='financial',
        output_dir=args.output_dir,
        random_state=args.random_state
    )
    
    # Prepare financial training data
    X_financial, y_financial = financial_trainer.prepare_data(
        data=financial_data,
        label_col=args.label_col,
        exclude_cols=args.exclude_cols,
        use_saved_preprocessing=not args.disable_saved_preprocessing
    )
    
    # Train financial models
    financial_trainer.train_models(
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
            exclude_cols=args.exclude_cols,
            use_saved_preprocessing=not args.disable_saved_preprocessing
        )
        
        # Evaluate financial models on test data
        financial_trainer.evaluate_on_test_data(
            X_test=X_financial_test,
            y_test=y_financial_test
        )
    
    logger.info("\n=== Training Complete ===")
    logger.info(f"Models saved to {args.output_dir}")
    
    # Log enhancement summary
    if HAS_FINANCIAL_PREPROCESSOR:
        logger.info("\n🚀 Enhanced Financial Preprocessing Features:")
        logger.info("  ✅ Consistent preprocessing parameters loaded from training")
        logger.info("  ✅ Comprehensive financial ratios (P&L/revenue, balance sheet/total assets)")
        logger.info("  ✅ 1-3 year growth metrics and trend indicators")
        logger.info("  ✅ Data quality validation and feature alignment")
        logger.info("  ✅ Proper ticker format handling (.L suffix)")
    else:
        logger.info("\n⚠️  Using standard preprocessing (financial_preprocessor.py not found)")

if __name__ == '__main__':
    main() 