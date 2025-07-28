#!/usr/bin/env python3
"""
Train Domain Models with Cross-Validation

This script trains domain-specific ML models with cross-validation
and evaluates their performance.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# ML libraries
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class DomainModelTrainer:
    """Base class for domain-specific model trainers"""
    
    def __init__(
        self,
        data_path: str,
        output_dir: str,
        label_col: str = 'label',
        exclude_cols: List[str] = None,
        models: List[str] = None,
        cv_folds: int = 5,
        scoring: str = 'f1_weighted',
        random_state: int = 42
    ):
        """Initialize the trainer
        
        Args:
            data_path: Path to the data CSV
            output_dir: Directory to save trained models
            label_col: Name of the label column
            exclude_cols: Columns to exclude from features
            models: Models to train
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for cross-validation
            random_state: Random state for reproducibility
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.label_col = label_col
        self.exclude_cols = exclude_cols or ['outperformance_10d', 'setup_id']
        self.models = models or ['random_forest', 'xgboost', 'logistic_regression']
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.domain_name = 'base'
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load data from CSV
            
        Returns:
            Tuple of (features, labels)
        """
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Separate features and labels
        exclude_cols = self.exclude_cols.copy()
        if self.label_col not in exclude_cols:
            exclude_cols.append(self.label_col)
            
        X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        y = df[self.label_col].copy()
        
        return X, y
    
    def train_models_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_X: Optional[pd.DataFrame] = None,
        test_y: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Train models with cross-validation
        
        Args:
            X: Features
            y: Labels
            test_X: Test features (optional)
            test_y: Test labels (optional)
            
        Returns:
            Dictionary of trained models and evaluation results
        """
        logger.info(f"Prepared {self.domain_name} data with {X.shape[1]} features and {X.shape[0]} samples")
        logger.info(f"Label distribution: {dict(y.value_counts().sort_index())}")
        
        results = {}
        
        for model_name in self.models:
            logger.info(f"Training {self.domain_name} {model_name} with {self.cv_folds}-fold cross-validation...")
            
            # Create model pipeline
            model = self._create_model(model_name)
            
            # For XGBoost, we need to handle the -1, 0, 1 labels
            if model_name == 'xgboost':
                # Convert labels from [-1, 0, 1] to [0, 1, 2]
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
                
                # Perform cross-validation
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                cv_results = cross_validate(
                    model,
                    X,
                    y_encoded,  # Use encoded labels for XGBoost
                    cv=cv,
                    scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                    return_estimator=True
                )
            else:
                # Perform cross-validation
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                cv_results = cross_validate(
                    model,
                    X,
                    y,
                    cv=cv,
                    scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                    return_estimator=True
                )
            
            # Log CV results
            logger.info(f"  - Mean Accuracy: {cv_results['test_accuracy'].mean():.4f} (±{cv_results['test_accuracy'].std():.4f})")
            logger.info(f"  - Mean Precision: {cv_results['test_precision_weighted'].mean():.4f} (±{cv_results['test_precision_weighted'].std():.4f})")
            logger.info(f"  - Mean Recall: {cv_results['test_recall_weighted'].mean():.4f} (±{cv_results['test_recall_weighted'].std():.4f})")
            logger.info(f"  - Mean F1 Score: {cv_results['test_f1_weighted'].mean():.4f} (±{cv_results['test_f1_weighted'].std():.4f})")
            
            # Get best model (by F1 score)
            best_idx = np.argmax(cv_results['test_f1_weighted'])
            best_model = cv_results['estimator'][best_idx]
            
            # Save model
            model_path = os.path.join(self.output_dir, f"{model_name}.pkl")
            pd.to_pickle(best_model, model_path)
            
            # Save feature importance plot if available
            if hasattr(best_model, 'feature_importances_') or (hasattr(best_model, 'named_steps') and hasattr(best_model.named_steps.get('classifier', None), 'feature_importances_')):
                self._plot_feature_importance(best_model, X.columns, model_name)
            
            # Evaluate on test set if provided
            if test_X is not None and test_y is not None:
                if model_name == 'xgboost':
                    # Convert test labels from [-1, 0, 1] to [0, 1, 2]
                    test_y_encoded = label_encoder.transform(test_y)
                    
                    # Make predictions
                    test_pred = best_model.predict(test_X)
                    
                    # Convert predictions back to original labels
                    test_pred = label_encoder.inverse_transform(test_pred)
                    
                    # Evaluate
                    test_accuracy = accuracy_score(test_y, test_pred)
                    test_precision = precision_score(test_y, test_pred, average='weighted')
                    test_recall = recall_score(test_y, test_pred, average='weighted')
                    test_f1 = f1_score(test_y, test_pred, average='weighted')
                else:
                    # Make predictions
                    test_pred = best_model.predict(test_X)
                    
                    # Evaluate
                    test_accuracy = accuracy_score(test_y, test_pred)
                    test_precision = precision_score(test_y, test_pred, average='weighted')
                    test_recall = recall_score(test_y, test_pred, average='weighted')
                    test_f1 = f1_score(test_y, test_pred, average='weighted')
                
                logger.info(f"  - Test Accuracy: {test_accuracy:.4f}")
                logger.info(f"  - Test Precision: {test_precision:.4f}")
                logger.info(f"  - Test Recall: {test_recall:.4f}")
                logger.info(f"  - Test F1 Score: {test_f1:.4f}")
                
                # Save confusion matrix
                cm = confusion_matrix(test_y, test_pred)
                self._plot_confusion_matrix(cm, model_name)
            
            # Store results
            results[model_name] = {
                'model': best_model,
                'cv_results': cv_results,
                'feature_names': X.columns.tolist()
            }
            
            if test_X is not None and test_y is not None:
                results[model_name]['test_accuracy'] = test_accuracy
                results[model_name]['test_precision'] = test_precision
                results[model_name]['test_recall'] = test_recall
                results[model_name]['test_f1'] = test_f1
                results[model_name]['test_predictions'] = test_pred
        
        return results
    
    def _create_model(self, model_name: str) -> Any:
        """Create a model pipeline
        
        Args:
            model_name: Name of the model to create
            
        Returns:
            Model pipeline
        """
        if model_name == 'random_forest':
            return Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('classifier', RandomForestClassifier(
                        n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                        random_state=self.random_state
                    ))
                ])
        elif model_name == 'xgboost':
            return Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                ('classifier', xgb.XGBClassifier(
                            n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                ))
            ])
        elif model_name == 'logistic_regression':
            return Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(
                    C=1.0,
                    penalty='l2',
                    solver='liblinear',
                    multi_class='ovr',
                        random_state=self.random_state
                    ))
                ])
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _plot_feature_importance(self, model: Any, feature_names: List[str], model_name: str) -> None:
        """Plot feature importance
        
        Args:
            model: Trained model
            feature_names: Feature names
            model_name: Name of the model
        """
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('classifier', None), 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
        else:
            return
        
        # Sort feature importances
        indices = np.argsort(importances)[::-1]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importances - {model_name.capitalize()}')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.join(self.output_dir, self.domain_name), exist_ok=True)
        plt.savefig(os.path.join(self.output_dir, self.domain_name, f'{model_name}_feature_importances.png'))
        logger.info(f"Saved feature importance plot to {os.path.join(self.output_dir, self.domain_name, f'{model_name}_feature_importances.png')}")
    
    def _plot_confusion_matrix(self, cm: np.ndarray, model_name: str) -> None:
        """Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            model_name: Name of the model
        """
        plt.figure(figsize=(8, 6))
        plt.title(f'Confusion Matrix - {model_name.capitalize()}')
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        
        classes = np.unique([-1, 0, 1])
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.join(self.output_dir, self.domain_name), exist_ok=True)
        plt.savefig(os.path.join(self.output_dir, self.domain_name, f'{model_name}_confusion_matrix.png'))
        logger.info(f"Saved confusion matrix plot to {os.path.join(self.output_dir, self.domain_name, f'{model_name}_confusion_matrix.png')}")


class TextModelTrainer(DomainModelTrainer):
    """Trainer for text domain models"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_name = 'text'


class FinancialModelTrainer(DomainModelTrainer):
    """Trainer for financial domain models"""
    
    def __init__(self, *args, disable_saved_preprocessing: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_name = 'financial'
        self.disable_saved_preprocessing = disable_saved_preprocessing
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load data from CSV with financial preprocessing
        
        Returns:
            Tuple of (features, labels)
        """
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Separate features and labels
        exclude_cols = self.exclude_cols.copy()
        if self.label_col not in exclude_cols:
            exclude_cols.append(self.label_col)
            
        X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        y = df[self.label_col].copy()
        
        # Apply saved preprocessing if available and not disabled
        if not self.disable_saved_preprocessing:
            try:
                from financial_preprocessor import FinancialPreprocessor
                preprocessor = FinancialPreprocessor()
                
                # Check if preprocessing parameters are available
                if hasattr(preprocessor, 'has_saved_parameters') and preprocessor.has_saved_parameters():
                    logger.info("Applying saved financial preprocessing parameters")
                    X = preprocessor.apply_preprocessing(X)
                else:
                    logger.warning("No saved preprocessing parameters found, using raw features")
            except ImportError:
                logger.warning("financial_preprocessor module not found, using raw features")
            except Exception as e:
                logger.warning(f"Error applying financial preprocessing: {e}")
        
        return X, y


def train_and_evaluate(
    text_data: Optional[str] = None,
    financial_data: Optional[str] = None,
    text_test_data: Optional[str] = None,
    financial_test_data: Optional[str] = None,
    output_dir: str = 'models',
    label_col: str = 'label',
    exclude_cols: List[str] = None,
    models: List[str] = None,
    cv_folds: int = 5,
    scoring: str = 'f1_weighted',
    random_state: int = 42,
    disable_saved_preprocessing: bool = False
) -> Dict[str, Any]:
    """Train and evaluate domain models
    
    Args:
        text_data: Path to text ML features CSV (optional)
        financial_data: Path to financial ML features CSV (optional)
        text_test_data: Path to text ML features test CSV (optional)
        financial_test_data: Path to financial ML features test CSV (optional)
        output_dir: Directory to save trained models
        label_col: Name of the label column
        exclude_cols: Columns to exclude from features
        models: Models to train
        cv_folds: Number of cross-validation folds
        scoring: Scoring metric for cross-validation
        random_state: Random state for reproducibility
        disable_saved_preprocessing: Disable loading saved preprocessing parameters for financial domain
        
    Returns:
        Dictionary of trained models and evaluation results
    """
    # Create output directories
    results = {}
    
    # Train text models if text data is provided
    if text_data:
        os.makedirs(os.path.join(output_dir, 'text'), exist_ok=True)
        
        # Initialize text trainer
        text_trainer = TextModelTrainer(
            data_path=text_data,
            output_dir=output_dir,
            label_col=label_col,
            exclude_cols=exclude_cols,
            models=models,
            cv_folds=cv_folds,
            scoring=scoring,
            random_state=random_state
        )
        
        # Load data
        X_text, y_text = text_trainer.load_data()
        
        # Load test data if provided
        test_X_text = None
        test_y_text = None
        
        if text_test_data:
            # Load test data for text domain
            test_df_text = pd.read_csv(text_test_data)
            exclude_cols_text = exclude_cols.copy() if exclude_cols else ['outperformance_10d', 'setup_id']
            if label_col not in exclude_cols_text:
                exclude_cols_text.append(label_col)
            test_X_text = test_df_text.drop(columns=[col for col in exclude_cols_text if col in test_df_text.columns])
            test_y_text = test_df_text[label_col].copy()
        
        # Train text models
        logger.info("Training text domain models...")
        text_results = text_trainer.train_models_with_cv(
            X_text,
            y_text,
            test_X_text,
            test_y_text
        )
        results['text'] = text_results
    
    # Train financial models if financial data is provided
    if financial_data:
        os.makedirs(os.path.join(output_dir, 'financial'), exist_ok=True)
        
        # Initialize financial trainer
        financial_trainer = FinancialModelTrainer(
            data_path=financial_data,
            output_dir=output_dir,
            label_col=label_col,
            exclude_cols=exclude_cols,
            models=models,
            cv_folds=cv_folds,
            scoring=scoring,
            random_state=random_state,
            disable_saved_preprocessing=disable_saved_preprocessing
        )
        
        # Load data
        X_financial, y_financial = financial_trainer.load_data()
        
        # Load test data if provided
        test_X_financial = None
        test_y_financial = None
        
        if financial_test_data:
            # Load test data for financial domain
            test_df_financial = pd.read_csv(financial_test_data)
            exclude_cols_financial = exclude_cols.copy() if exclude_cols else ['outperformance_10d', 'setup_id']
            if label_col not in exclude_cols_financial:
                exclude_cols_financial.append(label_col)
            test_X_financial = test_df_financial.drop(columns=[col for col in exclude_cols_financial if col in test_df_financial.columns])
            test_y_financial = test_df_financial[label_col].copy()
            
            # Apply financial preprocessing to test data if available and not disabled
            if not disable_saved_preprocessing:
                try:
                    from financial_preprocessor import FinancialPreprocessor
                    preprocessor = FinancialPreprocessor()
                    
                    # Check if preprocessing parameters are available
                    if hasattr(preprocessor, 'has_saved_parameters') and preprocessor.has_saved_parameters():
                        logger.info("Applying saved financial preprocessing parameters to test data")
                        test_X_financial = preprocessor.apply_preprocessing(test_X_financial)
                except ImportError:
                    logger.warning("financial_preprocessor module not found, using raw features for test data")
                except Exception as e:
                    logger.warning(f"Error applying financial preprocessing to test data: {e}")
        
        # Train financial models
        logger.info("Training financial domain models...")
        financial_results = financial_trainer.train_models_with_cv(
            X_financial,
            y_financial,
            test_X_financial,
            test_y_financial
        )
        results['financial'] = financial_results
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train domain-specific ML models with cross-validation')
    parser.add_argument('--text-data', help='Path to text ML features CSV')
    parser.add_argument('--financial-data', help='Path to financial ML features CSV')
    parser.add_argument('--text-test-data', help='Path to text ML features test CSV')
    parser.add_argument('--financial-test-data', help='Path to financial ML features test CSV')
    parser.add_argument('--output-dir', default='models', help='Directory to save trained models')
    parser.add_argument('--label-col', default='label', help='Name of the label column')
    parser.add_argument('--exclude-cols', nargs='+', default=['outperformance_10d', 'setup_id'], help='Columns to exclude from features')
    parser.add_argument('--models', nargs='+', default=['random_forest', 'xgboost', 'logistic_regression'], help='Models to train')
    parser.add_argument('--cv', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--scoring', default='f1_weighted', help='Scoring metric for cross-validation')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--disable-saved-preprocessing', action='store_true', help='Disable loading saved preprocessing parameters for financial domain')
    
    args = parser.parse_args()
    
    # Ensure at least one of text_data or financial_data is provided
    if not args.text_data and not args.financial_data:
        parser.error("At least one of --text-data or --financial-data must be provided")
    
    # Train and evaluate models
    results = train_and_evaluate(
        text_data=args.text_data,
        financial_data=args.financial_data,
        text_test_data=args.text_test_data,
        financial_test_data=args.financial_test_data,
        output_dir=args.output_dir,
        label_col=args.label_col,
        exclude_cols=args.exclude_cols,
        models=args.models,
        cv_folds=args.cv,
        scoring=args.scoring,
        random_state=args.random_state,
        disable_saved_preprocessing=args.disable_saved_preprocessing
    )
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main() 