#!/usr/bin/env python3
"""
3-Stage ML Pipeline for Text, Financial, and Ensemble Training

Stage 1: Train Text ML models (Random Forest, XGBoost, LightGBM, Logistic Regression)
Stage 2: Train Financial ML models (Random Forest, XGBoost, LightGBM, Logistic Regression)  
Stage 3: Train Ensemble meta-model using 8 prediction vectors from stages 1 & 2

Usage:
    python train_3stage_ml_pipeline.py --db-path path/to/database.db --output-dir models
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.base import clone

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from corrected_csv_feature_loader import CorrectedCSVFeatureLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThreeStageMLPipeline:
    """3-Stage ML Pipeline: Text ML -> Financial ML -> Ensemble ML"""
    
    def __init__(self, db_path: str, output_dir: str = "models", random_state: int = 42):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        
        # Create output directories
        self.text_model_dir = self.output_dir / "text"
        self.financial_model_dir = self.output_dir / "financial"
        self.ensemble_model_dir = self.output_dir / "ensemble"
        
        for dir_path in [self.text_model_dir, self.financial_model_dir, self.ensemble_model_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize feature loader
        self.feature_loader = CorrectedCSVFeatureLoader()
        
        # Define models
        self.model_configs = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                class_weight='balanced', random_state=random_state, n_jobs=-1
            ),
            'logistic_regression': LogisticRegression(
                C=1.0, class_weight='balanced', random_state=random_state, 
                max_iter=1000, solver='liblinear'
            )
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            self.model_configs['xgboost'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=random_state,
                n_jobs=-1, eval_metric='mlogloss', use_label_encoder=False
            )
        
        # Add LightGBM if available  
        if HAS_LIGHTGBM:
            self.model_configs['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                class_weight='balanced', random_state=random_state, 
                n_jobs=-1, verbose=-1
            )
            
        logger.info(f"Initialized pipeline with {len(self.model_configs)} models: {list(self.model_configs.keys())}")
        
    def train_stage1_text_models(self) -> Dict[str, Any]:
        """Stage 1: Train text-based ML models"""
        logger.info("🔤 Stage 1: Training Text-based ML Models")
        
        # Load text features
        X_text, y_text = self.feature_loader.load_text_features("training")
        if X_text is None:
            raise ValueError("Failed to load text features")
            
        logger.info(f"Text features shape: {X_text.shape}")
        logger.info(f"Label distribution: {pd.Series(y_text).value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y_text, test_size=0.2, random_state=self.random_state, stratify=y_text
        )
        
        # Train models and collect prediction vectors
        text_models = {}
        text_results = {}
        prediction_vectors_train = pd.DataFrame(index=X_train.index)
        prediction_vectors_test = pd.DataFrame(index=X_test.index)
        
        for model_name, model_config in self.model_configs.items():
            logger.info(f"  🎯 Training text {model_name}")
            
            # Create preprocessing pipeline
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', clone(model_config))
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Predictions and probabilities
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)
            y_proba_train = pipeline.predict_proba(X_train)
            y_proba_test = pipeline.predict_proba(X_test)
            
            # Store prediction vectors (probabilities for each class)
            for i, class_label in enumerate([0, 1, 2]):  # negative, neutral, positive
                prediction_vectors_train[f'text_{model_name}_prob_{class_label}'] = y_proba_train[:, i]
                prediction_vectors_test[f'text_{model_name}_prob_{class_label}'] = y_proba_test[:, i]
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
            
            logger.info(f"    ✅ Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            # Store results
            text_models[model_name] = pipeline
            text_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred_test,
                'y_proba': y_proba_test
            }
            
            # Save model
            model_path = self.text_model_dir / f"text_{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(pipeline, f)
            logger.info(f"    💾 Saved model to {model_path}")
            
        # Generate confusion matrix plot
        self._plot_confusion_matrices(y_test, text_results, "Text Models", self.text_model_dir)
        
        # Save prediction vectors
        prediction_vectors_path = self.text_model_dir / "text_prediction_vectors.pkl"
        with open(prediction_vectors_path, 'wb') as f:
            pickle.dump({
                'train': prediction_vectors_train,
                'test': prediction_vectors_test,
                'y_train': y_train,
                'y_test': y_test
            }, f)
        
        return {
            'models': text_models,
            'results': text_results,
            'prediction_vectors_train': prediction_vectors_train,
            'prediction_vectors_test': prediction_vectors_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(X_text.columns)
        }
        
    def train_stage2_financial_models(self) -> Dict[str, Any]:
        """Stage 2: Train financial-based ML models"""
        logger.info("💰 Stage 2: Training Financial-based ML Models")
        
        # Load financial features
        X_financial, y_financial = self.feature_loader.load_financial_features("training")
        if X_financial is None:
            raise ValueError("Failed to load financial features")
            
        logger.info(f"Financial features shape: {X_financial.shape}")
        logger.info(f"Label distribution: {pd.Series(y_financial).value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_financial, y_financial, test_size=0.2, random_state=self.random_state, stratify=y_financial
        )
        
        # Train models and collect prediction vectors
        financial_models = {}
        financial_results = {}
        prediction_vectors_train = pd.DataFrame(index=X_train.index)
        prediction_vectors_test = pd.DataFrame(index=X_test.index)
        
        for model_name, model_config in self.model_configs.items():
            logger.info(f"  🎯 Training financial {model_name}")
            
            # Create preprocessing pipeline
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', clone(model_config))
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Predictions and probabilities
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)
            y_proba_train = pipeline.predict_proba(X_train)
            y_proba_test = pipeline.predict_proba(X_test)
            
            # Store prediction vectors (probabilities for each class)
            for i, class_label in enumerate([0, 1, 2]):  # negative, neutral, positive
                prediction_vectors_train[f'financial_{model_name}_prob_{class_label}'] = y_proba_train[:, i]
                prediction_vectors_test[f'financial_{model_name}_prob_{class_label}'] = y_proba_test[:, i]
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
            
            logger.info(f"    ✅ Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            # Store results
            financial_models[model_name] = pipeline
            financial_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred_test,
                'y_proba': y_proba_test
            }
            
            # Save model
            model_path = self.financial_model_dir / f"financial_{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(pipeline, f)
            logger.info(f"    💾 Saved model to {model_path}")
            
        # Generate confusion matrix plot
        self._plot_confusion_matrices(y_test, financial_results, "Financial Models", self.financial_model_dir)
        
        # Save prediction vectors
        prediction_vectors_path = self.financial_model_dir / "financial_prediction_vectors.pkl"
        with open(prediction_vectors_path, 'wb') as f:
            pickle.dump({
                'train': prediction_vectors_train,
                'test': prediction_vectors_test,
                'y_train': y_train,
                'y_test': y_test
            }, f)
        
        return {
            'models': financial_models,
            'results': financial_results,
            'prediction_vectors_train': prediction_vectors_train,
            'prediction_vectors_test': prediction_vectors_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(X_financial.columns)
        }
        
    def train_stage3_ensemble_model(self, text_stage_results: Dict, financial_stage_results: Dict) -> Dict[str, Any]:
        """Stage 3: Train ensemble meta-model using prediction vectors from stages 1 & 2"""
        logger.info("🏆 Stage 3: Training Ensemble Meta-Model")
        
        # Get common setup IDs between text and financial data
        common_ids = self.feature_loader.get_common_setup_ids()
        if len(common_ids) < 50:  # Need minimum data for ensemble training
            raise ValueError(f"Not enough common setup IDs ({len(common_ids)}) for ensemble training")
            
        logger.info(f"Using {len(common_ids)} common setup IDs for ensemble training")
        
        # Load both text and financial features to get common indices
        X_text, y_text = self.feature_loader.load_text_features("training")
        X_financial, y_financial = self.feature_loader.load_financial_features("training")
        
        # Verify labels are consistent
        if not np.array_equal(y_text, y_financial):
            logger.warning("Labels differ between text and financial data, using text labels")
            
        # Split data for ensemble training
        X_train_indices, X_test_indices, y_train, y_test = train_test_split(
            range(len(common_ids)), y_text, test_size=0.2, 
            random_state=self.random_state, stratify=y_text
        )
        
        # Get prediction vectors for common setup IDs
        text_pred_train = pd.DataFrame(index=X_train_indices)
        text_pred_test = pd.DataFrame(index=X_test_indices)
        financial_pred_train = pd.DataFrame(index=X_train_indices)
        financial_pred_test = pd.DataFrame(index=X_test_indices)
        
        # Generate prediction vectors using trained models
        for model_name, model in text_stage_results['models'].items():
            # Predict on common data
            X_text_train = X_text.iloc[X_train_indices]
            X_text_test = X_text.iloc[X_test_indices]
            
            y_proba_train = model.predict_proba(X_text_train)
            y_proba_test = model.predict_proba(X_text_test)
            
            for i, class_label in enumerate([0, 1, 2]):
                text_pred_train[f'text_{model_name}_prob_{class_label}'] = y_proba_train[:, i]
                text_pred_test[f'text_{model_name}_prob_{class_label}'] = y_proba_test[:, i]
                
        for model_name, model in financial_stage_results['models'].items():
            # Predict on common data
            X_financial_train = X_financial.iloc[X_train_indices]
            X_financial_test = X_financial.iloc[X_test_indices]
            
            y_proba_train = model.predict_proba(X_financial_train)
            y_proba_test = model.predict_proba(X_financial_test)
            
            for i, class_label in enumerate([0, 1, 2]):
                financial_pred_train[f'financial_{model_name}_prob_{class_label}'] = y_proba_train[:, i]
                financial_pred_test[f'financial_{model_name}_prob_{class_label}'] = y_proba_test[:, i]
        
        # Combine prediction vectors from both domains
        ensemble_X_train = pd.concat([text_pred_train, financial_pred_train], axis=1)
        ensemble_X_test = pd.concat([text_pred_test, financial_pred_test], axis=1)
        
        logger.info(f"Ensemble feature shape: {ensemble_X_train.shape} (8 models × 3 classes = {ensemble_X_train.shape[1]} features)")
        
        # Train ensemble meta-models
        ensemble_models = {}
        ensemble_results = {}
        
        for model_name, model_config in self.model_configs.items():
            logger.info(f"  🎯 Training ensemble {model_name}")
            
            # Create simple pipeline (no imputation needed for prediction vectors)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', clone(model_config))
            ])
            
            # Train ensemble model
            pipeline.fit(ensemble_X_train, y_train)
            
            # Predictions
            y_pred_test = pipeline.predict(ensemble_X_test)
            y_proba_test = pipeline.predict_proba(ensemble_X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, ensemble_X_train, y_train, cv=5, scoring='accuracy')
            
            logger.info(f"    ✅ Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            # Store results
            ensemble_models[model_name] = pipeline
            ensemble_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred_test,
                'y_proba': y_proba_test
            }
            
            # Save model
            model_path = self.ensemble_model_dir / f"ensemble_{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(pipeline, f)
            logger.info(f"    💾 Saved ensemble model to {model_path}")
            
        # Generate confusion matrix plot
        self._plot_confusion_matrices(y_test, ensemble_results, "Ensemble Models", self.ensemble_model_dir)
        
        # Save ensemble data
        ensemble_data_path = self.ensemble_model_dir / "ensemble_data.pkl"
        with open(ensemble_data_path, 'wb') as f:
            pickle.dump({
                'X_train': ensemble_X_train,
                'X_test': ensemble_X_test,
                'y_train': y_train,
                'y_test': y_test,
                'common_ids': common_ids,
                'feature_names': list(ensemble_X_train.columns)
            }, f)
        
        return {
            'models': ensemble_models,
            'results': ensemble_results,
            'X_train': ensemble_X_train,
            'X_test': ensemble_X_test,
            'y_train': y_train,
            'y_test': y_test,
            'common_ids': common_ids
        }
        
    def _plot_confusion_matrices(self, y_true: np.ndarray, results: Dict, title: str, save_dir: Path):
        """Plot confusion matrices for all models"""
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
            
        class_names = ['Negative', 'Neutral', 'Positive']
        
        for i, (model_name, result) in enumerate(results.items()):
            cm = confusion_matrix(y_true, result['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[i])
            axes[i].set_title(f'{model_name}\nAcc: {result["accuracy"]:.3f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            
        plt.suptitle(f'{title} - Confusion Matrices')
        plt.tight_layout()
        plt.savefig(save_dir / f'{title.lower().replace(" ", "_")}_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete 3-stage ML pipeline"""
        logger.info("🚀 Starting Complete 3-Stage ML Pipeline")
        
        # Stage 1: Text Models
        text_results = self.train_stage1_text_models()
        
        # Stage 2: Financial Models
        financial_results = self.train_stage2_financial_models()
        
        # Stage 3: Ensemble Models
        ensemble_results = self.train_stage3_ensemble_model(text_results, financial_results)
        
        # Generate final report
        self._generate_final_report(text_results, financial_results, ensemble_results)
        
        logger.info("✅ Complete 3-Stage ML Pipeline finished successfully!")
        
        return {
            'text': text_results,
            'financial': financial_results,
            'ensemble': ensemble_results
        }
        
    def _generate_final_report(self, text_results: Dict, financial_results: Dict, ensemble_results: Dict):
        """Generate comprehensive final report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"3stage_ml_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("3-STAGE ML PIPELINE TRAINING REPORT\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Stage 1: Text Results
            f.write("STAGE 1: TEXT-BASED ML RESULTS\n")
            f.write("-" * 30 + "\n")
            for model_name, result in text_results['results'].items():
                f.write(f"{model_name.upper()}:\n")
                f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"  Precision: {result['precision']:.4f}\n")
                f.write(f"  CV Score: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})\n\n")
                
            # Stage 2: Financial Results
            f.write("STAGE 2: FINANCIAL ML RESULTS\n")
            f.write("-" * 30 + "\n")
            for model_name, result in financial_results['results'].items():
                f.write(f"{model_name.upper()}:\n")
                f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"  Precision: {result['precision']:.4f}\n")
                f.write(f"  CV Score: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})\n\n")
                
            # Stage 3: Ensemble Results
            f.write("STAGE 3: ENSEMBLE ML RESULTS\n")
            f.write("-" * 30 + "\n")
            for model_name, result in ensemble_results['results'].items():
                f.write(f"{model_name.upper()}:\n")
                f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"  Precision: {result['precision']:.4f}\n")
                f.write(f"  CV Score: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})\n\n")
                
            # Best models
            f.write("BEST PERFORMING MODELS\n")
            f.write("-" * 20 + "\n")
            
            # Find best models
            best_text = max(text_results['results'].items(), key=lambda x: x[1]['accuracy'])
            best_financial = max(financial_results['results'].items(), key=lambda x: x[1]['accuracy'])
            best_ensemble = max(ensemble_results['results'].items(), key=lambda x: x[1]['accuracy'])
            
            f.write(f"Best Text Model: {best_text[0]} (Accuracy: {best_text[1]['accuracy']:.4f})\n")
            f.write(f"Best Financial Model: {best_financial[0]} (Accuracy: {best_financial[1]['accuracy']:.4f})\n")
            f.write(f"Best Ensemble Model: {best_ensemble[0]} (Accuracy: {best_ensemble[1]['accuracy']:.4f})\n")
            
        logger.info(f"📄 Final report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='3-Stage ML Pipeline Training')
    parser.add_argument('--db-path', required=True, help='Path to DuckDB database')
    parser.add_argument('--output-dir', default='models', help='Output directory for models')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = ThreeStageMLPipeline(
        db_path=args.db_path,
        output_dir=args.output_dir,
        random_state=args.random_state
    )
    
    try:
        results = pipeline.run_complete_pipeline()
        logger.info("🎉 Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        raise


if __name__ == '__main__':
    main() 