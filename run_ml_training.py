#!/usr/bin/env python3
"""
Run ML training pipeline using processed feature tables.
This script:
1. Loads data from processed CSV files
2. Trains three-stage ML pipeline (text → financial → ensemble)
3. Saves models and generates reports
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from ml.unified_pipeline.unified_ml_pipeline import EnhancedUnifiedMLPipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_training_data(text_features_path: str, financial_features_path: str):
    """Load training data from processed CSV files"""
    logger.info("📊 Loading training data from CSV files...")
    
    # Load features
    text_df = pd.read_csv(text_features_path)
    financial_df = pd.read_csv(financial_features_path)
    
    # Basic data quality checks
    logger.info("\n📊 Data Quality Report:")
    logger.info(f"Text Features: {text_df.shape[1]-2} features, {text_df.shape[0]} samples")
    logger.info(f"Financial Features: {financial_df.shape[1]-2} features, {financial_df.shape[0]} samples")
    
    # Check label distribution
    logger.info("\nLabel Distribution:")
    label_dist = text_df['label'].value_counts()
    for label, count in label_dist.items():
        logger.info(f"  - Class {label}: {count} samples ({count/len(text_df)*100:.1f}%)")
    
    # Extract features and labels
    text_features = text_df.drop(['setup_id', 'label'], axis=1)
    financial_features = financial_df.drop(['setup_id', 'label'], axis=1)
    labels = text_df['label'].values  # Convert to numpy array
    
    return text_features, financial_features, labels

def train_models(X_train, y_train, X_test, y_test, model_type='text'):
    """Train models and calculate metrics"""
    logger.info(f"Training {model_type} models...")
    
    # Initialize models
    from ml.unified_pipeline.models.random_forest_model import RandomForestModel
    from ml.unified_pipeline.models.xgboost_model import XGBoostModel
    from ml.unified_pipeline.models.lightgbm_model import LightGBMModel
    from ml.unified_pipeline.models.logistic_regression_model import LogisticRegressionModel
    
    models = {
        'rf': RandomForestModel(),
        'xgb': XGBoostModel(),
        'lgb': LightGBMModel(),
        'logreg': LogisticRegressionModel()
    }
    
    # Train models
    predictions = {}
    feature_importance = pd.DataFrame()
    cv_metrics = {}
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Convert to numpy arrays
        X_train_np = X_train if isinstance(X_train, np.ndarray) else X_train.values
        X_test_np = X_test if isinstance(X_test, np.ndarray) else X_test.values
        y_train_np = y_train if isinstance(y_train, np.ndarray) else y_train.values
        y_test_np = y_test if isinstance(y_test, np.ndarray) else y_test.values
        
        # Cross-validation metrics
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        # Train and evaluate on each fold
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_np, y_train_np)):
            X_fold_train = X_train_np[train_idx]
            y_fold_train = y_train_np[train_idx]
            X_fold_val = X_train_np[val_idx]
            y_fold_val = y_train_np[val_idx]
            
            # Train model on fold
            history = model.train_with_early_stopping(
                X_fold_train, y_fold_train,
                X_fold_val, y_fold_val,
                list(range(X_train_np.shape[1]))  # Feature names as indices
            )
            
            # Get predictions for validation set
            fold_preds = model.predict_proba(X_fold_val)
            fold_pred_labels = np.argmax(fold_preds, axis=1)
            
            # Calculate metrics for fold
            cv_scores['accuracy'].append(accuracy_score(y_fold_val, fold_pred_labels))
            cv_scores['precision'].append(precision_score(y_fold_val, fold_pred_labels, average='weighted'))
            cv_scores['recall'].append(recall_score(y_fold_val, fold_pred_labels, average='weighted'))
            cv_scores['f1'].append(f1_score(y_fold_val, fold_pred_labels, average='weighted'))
        
        # Train final model on full training set
        history = model.train_with_early_stopping(
            X_train_np, y_train_np,
            X_test_np, y_test_np,
            list(range(X_train_np.shape[1]))  # Feature names as indices
        )
        
        # Get predictions for test set
        predictions[name] = model.predict_proba(X_test_np)
        
        # Get feature importance
        importance = model.get_feature_importance()
        if importance is not None:
            feature_importance[name] = importance
        
        # Store CV metrics
        cv_metrics[name] = {
            metric: {
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
            for metric, scores in cv_scores.items()
        }
    
    # Calculate metrics on test set
    metrics = {}
    for name in models.keys():
        pred_labels = np.argmax(predictions[name], axis=1)
        metrics[name] = {
            'accuracy': accuracy_score(y_test, pred_labels),
            'precision': precision_score(y_test, pred_labels, average='weighted'),
            'recall': recall_score(y_test, pred_labels, average='weighted'),
            'f1': f1_score(y_test, pred_labels, average='weighted')
        }
    
    # Calculate average predictions
    avg_pred = np.mean([pred for pred in predictions.values()], axis=0)
    pred_labels = np.argmax(avg_pred, axis=1)
    
    return {
        'predictions': predictions,
        'feature_importance': feature_importance.mean(axis=1).sort_values(ascending=False) if len(feature_importance) > 0 else pd.Series(),
        'y_true': y_test,
        'y_pred': pred_labels,
        'metrics': metrics,
        'cv_metrics': cv_metrics,  # Add cross-validation metrics
        'model_metrics': metrics  # Keep both for backward compatibility
    }

def train_ensemble(text_predictions, financial_predictions, y_test):
    """Train ensemble model"""
    logger.info("Training ensemble model...")
    
    # Create enhanced features
    def create_enhanced_features(predictions_dict):
        n_samples = next(iter(predictions_dict.values())).shape[0]
        n_classes = next(iter(predictions_dict.values())).shape[1]
        
        # Model agreement features
        agreements = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            votes = np.sum([preds[:, i] > 0.5 for preds in predictions_dict.values()], axis=0)
            agreements[:, i] = votes / len(predictions_dict)
        
        # Confidence features
        confidence = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            conf = np.mean([preds[:, i] for preds in predictions_dict.values()], axis=0)
            confidence[:, i] = conf
        
        # Confidence spread features
        spreads = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            spread = np.std([preds[:, i] for preds in predictions_dict.values()], axis=0)
            spreads[:, i] = spread
        
        return np.hstack([
            np.hstack([pred for pred in predictions_dict.values()]),
            agreements,
            confidence,
            spreads
        ])
    
    # Create enhanced features
    text_ensemble_features = create_enhanced_features(text_predictions)
    financial_ensemble_features = create_enhanced_features(financial_predictions)
    
    # Domain agreement features
    n_classes = next(iter(text_predictions.values())).shape[1]
    domain_agreements = np.zeros((text_ensemble_features.shape[0], n_classes))
    for i in range(n_classes):
        text_conf = np.mean([pred[:, i] for pred in text_predictions.values()], axis=0)
        financial_conf = np.mean([pred[:, i] for pred in financial_predictions.values()], axis=0)
        domain_agreements[:, i] = 1 - np.abs(text_conf - financial_conf)
    
    # Combine features
    ensemble_features = np.hstack([
        text_ensemble_features,
        financial_ensemble_features,
        domain_agreements
    ])
    
    # Train ensemble model
    from ml.unified_pipeline.models.xgboost_model import XGBoostModel
    ensemble_model = XGBoostModel()
    
    # Split ensemble data
    ensemble_size = int(ensemble_features.shape[0] * 0.8)
    X_ensemble_train = ensemble_features[:ensemble_size]
    X_ensemble_val = ensemble_features[ensemble_size:]
    y_ensemble_train = y_test[:ensemble_size]
    y_ensemble_val = y_test[ensemble_size:]
    
    # Train ensemble
    ensemble_history = ensemble_model.train_with_early_stopping(
        X_ensemble_train,
        y_ensemble_train,
        X_ensemble_val,
        y_ensemble_val,
        list(range(ensemble_features.shape[1]))  # Feature names as indices
    )
    
    # Get ensemble predictions
    ensemble_predictions = ensemble_model.predict_proba(ensemble_features)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    ensemble_pred_labels = np.argmax(ensemble_predictions, axis=1)
    ensemble_metrics = {
        'accuracy': accuracy_score(y_test, ensemble_pred_labels),
        'precision': precision_score(y_test, ensemble_pred_labels, average='weighted'),
        'recall': recall_score(y_test, ensemble_pred_labels, average='weighted'),
        'f1': f1_score(y_test, ensemble_pred_labels, average='weighted')
    }
    
    return {
        'predictions': ensemble_predictions,
        'y_true': y_test,
        'y_pred': ensemble_pred_labels,
        'metrics': {'ensemble': ensemble_metrics},
        'model_metrics': {'ensemble': ensemble_metrics}
    }

def main():
    """Run ML training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ML training pipeline')
    parser.add_argument('--text-features', default='data/ml_training/text_ml_features_training_20250727_040217.csv',
                       help='Path to text features CSV')
    parser.add_argument('--financial-features', default='data/ml_training/financial_ml_features_training_20250727_040217.csv',
                       help='Path to financial features CSV')
    parser.add_argument('--save-dir', default='ml/unified_pipeline/saved_states',
                       help='Directory to save models and reports')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    
    args = parser.parse_args()
    
    try:
        # Load data from CSV files
        text_features, financial_features, labels = load_training_data(
            args.text_features,
            args.financial_features
        )
        
        # Split data
        X_text_train, X_text_test, X_fin_train, X_fin_test, y_train, y_test = train_test_split(
            text_features,
            financial_features,
            labels,
            test_size=args.test_size,
            random_state=42,
            stratify=labels
        )
        
        # Train models
        text_results = train_models(
            X_text_train.values, y_train, 
            X_text_test.values, y_test,
            model_type='text'
        )
        financial_results = train_models(
            X_fin_train.values, y_train, 
            X_fin_test.values, y_test,
            model_type='financial'
        )
        ensemble_results = train_ensemble(
            text_results['predictions'], 
            financial_results['predictions'], 
            y_test
        )
        
        # Initialize pipeline for saving
        pipeline = EnhancedUnifiedMLPipeline(
            save_dir=args.save_dir
        )
        
        # Store results
        pipeline.text_results = text_results
        pipeline.financial_results = financial_results
        pipeline.ensemble_results = ensemble_results
        
        # Generate reports
        pipeline.visualization.generate_comprehensive_report(
            pipeline.text_results,
            pipeline.financial_results,
            pipeline.ensemble_results,
            ['negative', 'neutral', 'positive']  # Class names
        )
        
        # Save pipeline state
        pipeline.save_pipeline("latest")
        
        logger.info("\n🎉 Training Complete!")
        logger.info(f"Models and reports saved in: {args.save_dir}")
        logger.info("\nText ML Results:")
        for model, metrics in text_results['metrics'].items():
            logger.info(f"  {model}:")
            logger.info(f"    Accuracy: {metrics['accuracy']:.3f}")
            logger.info(f"    Precision: {metrics['precision']:.3f}")
            logger.info(f"    Recall: {metrics['recall']:.3f}")
            logger.info(f"    F1: {metrics['f1']:.3f}")
        
        logger.info("\nFinancial ML Results:")
        for model, metrics in financial_results['metrics'].items():
            logger.info(f"  {model}:")
            logger.info(f"    Accuracy: {metrics['accuracy']:.3f}")
            logger.info(f"    Precision: {metrics['precision']:.3f}")
            logger.info(f"    Recall: {metrics['recall']:.3f}")
            logger.info(f"    F1: {metrics['f1']:.3f}")
        
        logger.info("\nEnsemble ML Results:")
        metrics = ensemble_results['metrics']['ensemble']
        logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"  Precision: {metrics['precision']:.3f}")
        logger.info(f"  Recall: {metrics['recall']:.3f}")
        logger.info(f"  F1: {metrics['f1']:.3f}")
        
        # Save results summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(args.save_dir) / "reports" / f"training_results_{timestamp}.txt"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, "w") as f:
            f.write("ML Training Results Summary\n")
            f.write("=========================\n\n")
            f.write(f"Training completed at: {datetime.now()}\n")
            f.write(f"Data sizes:\n")
            f.write(f"  Text features: {text_features.shape}\n")
            f.write(f"  Financial features: {financial_features.shape}\n")
            f.write(f"  Labels: {len(labels)}\n\n")
            
            f.write("Text ML Results:\n")
            for model, metrics in text_results['metrics'].items():
                f.write(f"  {model}:\n")
                for metric, value in metrics.items():
                    f.write(f"    {metric}: {value:.3f}\n")
            
            f.write("\nFinancial ML Results:\n")
            for model, metrics in financial_results['metrics'].items():
                f.write(f"  {model}:\n")
                for metric, value in metrics.items():
                    f.write(f"    {metric}: {value:.3f}\n")
            
            f.write("\nEnsemble ML Results:\n")
            metrics = ensemble_results['metrics']['ensemble']
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.3f}\n")
        
        logger.info(f"\nResults summary saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 