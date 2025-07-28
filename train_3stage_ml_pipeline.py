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
from scipy.stats import pearsonr

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
    
    def __init__(self, db_path: str = None, output_dir: str = "models", random_state: int = 42):
        # db_path is kept for backward compatibility but not used with CSV loader
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
        
        # Class information for reporting
        self.class_names = ['Negative', 'Neutral', 'Positive']
        self.class_colors = ['#d62728', '#ff7f0e', '#2ca02c']  # Red, Orange, Green
        
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
                n_jobs=-1, eval_metric='mlogloss'
                # Removed use_label_encoder=False which is deprecated
            )
        
        # Add LightGBM if available  
        if HAS_LIGHTGBM:
            self.model_configs['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                class_weight='balanced', random_state=random_state, 
                n_jobs=-1, verbose=-1,
                importance_type='gain'  # Add importance_type to reduce warnings
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
            
            # Create enhanced preprocessing pipeline with robust handling
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median', missing_values=np.nan)),
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
            
            # Comprehensive evaluation
            eval_results = self.evaluate_model(pipeline, X_test, y_test, model_name)
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
            
            # Log results
            logger.info(f"    ✅ Accuracy: {eval_results['accuracy']:.4f}, Precision: {eval_results['precision_macro']:.4f}, CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            logger.info(f"    📊 Class-wise Precision: Neg={eval_results['precision_per_class'][0]:.3f}, Neu={eval_results['precision_per_class'][1]:.3f}, Pos={eval_results['precision_per_class'][2]:.3f}")
            
            # Store results
            text_models[model_name] = pipeline
            text_results[model_name] = {
                **eval_results,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
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
            
            # Create enhanced preprocessing pipeline with robust handling
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median', missing_values=np.nan)),
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
            
            # Comprehensive evaluation
            eval_results = self.evaluate_model(pipeline, X_test, y_test, model_name)
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
            
            # Log results
            logger.info(f"    ✅ Accuracy: {eval_results['accuracy']:.4f}, Precision: {eval_results['precision_macro']:.4f}, CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            logger.info(f"    📊 Class-wise Precision: Neg={eval_results['precision_per_class'][0]:.3f}, Neu={eval_results['precision_per_class'][1]:.3f}, Pos={eval_results['precision_per_class'][2]:.3f}")
            
            # Store results
            financial_models[model_name] = pipeline
            financial_results[model_name] = {
                **eval_results,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
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
        
    def analyze_prediction_correlation(self, text_preds, fund_preds, text_probs, fund_probs):
        """Analyze correlation between text and fundamentals predictions"""
        logger.info("\n🔍 PREDICTION CORRELATION ANALYSIS")
        
        # Prediction agreement
        agreement = (text_preds == fund_preds).mean()
        logger.info(f"📊 Prediction Agreement: {agreement:.3f} ({agreement*100:.1f}%)")
        
        # Class-wise agreement
        for i, class_name in enumerate(self.class_names):
            text_class_mask = (text_preds == i)
            fund_class_mask = (fund_preds == i)
            
            if text_class_mask.sum() > 0 and fund_class_mask.sum() > 0:
                class_agreement = (text_class_mask & fund_class_mask).sum() / max(text_class_mask.sum(), fund_class_mask.sum())
                logger.info(f"   {class_name} class agreement: {class_agreement:.3f}")
        
        # Probability correlations
        logger.info(f"\n📊 Probability Correlations:")
        correlation_results = {}
        
        for i, class_name in enumerate(self.class_names):
            try:
                corr, p_value = pearsonr(text_probs[:, i], fund_probs[:, i])
                logger.info(f"   {class_name} probabilities: r={corr:.3f} (p={p_value:.3f})")
                correlation_results[f"{class_name}_correlation"] = corr
                correlation_results[f"{class_name}_p_value"] = p_value
            except Exception as e:
                logger.warning(f"   Could not calculate correlation for {class_name}: {e}")
        
        # Overall correlation of max probabilities (confidence)
        text_confidence = np.max(text_probs, axis=1)
        fund_confidence = np.max(fund_probs, axis=1)
        
        try:
            confidence_corr, confidence_p = pearsonr(text_confidence, fund_confidence)
            logger.info(f"   Confidence correlation: r={confidence_corr:.3f} (p={confidence_p:.3f})")
            correlation_results["confidence_correlation"] = confidence_corr
            correlation_results["confidence_p_value"] = confidence_p
        except Exception as e:
            logger.warning(f"   Could not calculate confidence correlation: {e}")
        
        # Disagreement analysis
        disagreement_mask = (text_preds != fund_preds)
        disagreement_rate = disagreement_mask.mean()
        logger.info(f"\n🤔 Disagreement Analysis:")
        logger.info(f"   Disagreement rate: {disagreement_rate:.3f} ({disagreement_rate*100:.1f}%)")
        
        if disagreement_rate > 0:
            # Where text says positive but fundamentals says negative (and vice versa)
            text_pos_fund_neg = ((text_preds == 2) & (fund_preds == 0)).sum()
            text_neg_fund_pos = ((text_preds == 0) & (fund_preds == 2)).sum()
            logger.info(f"   Text Positive, Fund Negative: {text_pos_fund_neg}")
            logger.info(f"   Text Negative, Fund Positive: {text_neg_fund_pos}")
        
        correlation_results.update({
            'prediction_agreement': agreement,
            'disagreement_rate': disagreement_rate,
        })
        
        return correlation_results
        
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
        
        # Get predictions for correlation analysis
        best_text_model = max(text_stage_results['results'].items(), key=lambda x: x[1]['accuracy'])[0]
        best_financial_model = max(financial_stage_results['results'].items(), key=lambda x: x[1]['accuracy'])[0]
        
        text_preds = text_stage_results['models'][best_text_model].predict(X_text_test)
        text_probs = text_stage_results['models'][best_text_model].predict_proba(X_text_test)
        fund_preds = financial_stage_results['models'][best_financial_model].predict(X_financial_test)
        fund_probs = financial_stage_results['models'][best_financial_model].predict_proba(X_financial_test)
        
        # Analyze prediction correlation
        correlation_results = self.analyze_prediction_correlation(text_preds, fund_preds, text_probs, fund_probs)
        
        # Combine prediction vectors from both domains
        ensemble_X_train = pd.concat([text_pred_train, financial_pred_train], axis=1)
        ensemble_X_test = pd.concat([text_pred_test, financial_pred_test], axis=1)
        
        # Add richer ensemble features
        # Calculate confidence measures (max probability)
        for model_name in self.model_configs.keys():
            if f'text_{model_name}_prob_0' in ensemble_X_train:
                # Text model confidence
                ensemble_X_train[f'text_{model_name}_confidence'] = ensemble_X_train[[
                    f'text_{model_name}_prob_0', 
                    f'text_{model_name}_prob_1', 
                    f'text_{model_name}_prob_2'
                ]].max(axis=1)
                
                ensemble_X_test[f'text_{model_name}_confidence'] = ensemble_X_test[[
                    f'text_{model_name}_prob_0', 
                    f'text_{model_name}_prob_1', 
                    f'text_{model_name}_prob_2'
                ]].max(axis=1)
                
                # Financial model confidence
                ensemble_X_train[f'financial_{model_name}_confidence'] = ensemble_X_train[[
                    f'financial_{model_name}_prob_0', 
                    f'financial_{model_name}_prob_1', 
                    f'financial_{model_name}_prob_2'
                ]].max(axis=1)
                
                ensemble_X_test[f'financial_{model_name}_confidence'] = ensemble_X_test[[
                    f'financial_{model_name}_prob_0', 
                    f'financial_{model_name}_prob_1', 
                    f'financial_{model_name}_prob_2'
                ]].max(axis=1)
                
                # Confidence difference
                ensemble_X_train[f'{model_name}_confidence_diff'] = np.abs(
                    ensemble_X_train[f'text_{model_name}_confidence'] - 
                    ensemble_X_train[f'financial_{model_name}_confidence']
                )
                
                ensemble_X_test[f'{model_name}_confidence_diff'] = np.abs(
                    ensemble_X_test[f'text_{model_name}_confidence'] - 
                    ensemble_X_test[f'financial_{model_name}_confidence']
                )
                
                # Add agreement measures (same as in prediction pipeline)
                for i in range(3):  # For each class (0, 1, 2)
                    text_prob_train = ensemble_X_train[f'text_{model_name}_prob_{i}']
                    fund_prob_train = ensemble_X_train[f'financial_{model_name}_prob_{i}']
                    ensemble_X_train[f'{model_name}_class{i}_agreement'] = 1 - np.abs(text_prob_train - fund_prob_train)
                    
                    text_prob_test = ensemble_X_test[f'text_{model_name}_prob_{i}']
                    fund_prob_test = ensemble_X_test[f'financial_{model_name}_prob_{i}']
                    ensemble_X_test[f'{model_name}_class{i}_agreement'] = 1 - np.abs(text_prob_test - fund_prob_test)
        
        logger.info(f"Ensemble feature shape: {ensemble_X_train.shape} features")
        
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
            
            # Comprehensive evaluation
            eval_results = self.evaluate_model(pipeline, ensemble_X_test, y_test, model_name)
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, ensemble_X_train, y_train, cv=5, scoring='accuracy')
            
            # Log results
            logger.info(f"    ✅ Accuracy: {eval_results['accuracy']:.4f}, Precision: {eval_results['precision_macro']:.4f}, CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            logger.info(f"    📊 Class-wise Precision: Neg={eval_results['precision_per_class'][0]:.3f}, Neu={eval_results['precision_per_class'][1]:.3f}, Pos={eval_results['precision_per_class'][2]:.3f}")
            
            # Store results
            ensemble_models[model_name] = pipeline
            ensemble_results[model_name] = {
                **eval_results,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
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
        
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation with class-specific metrics"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Class-wise precision and recall
        precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(model[-1], 'feature_importances_'):
            feature_importance = model[-1].feature_importances_
        elif hasattr(model[-1], 'coef_'):
            # For logistic regression, use absolute coefficients
            feature_importance = np.abs(model[-1].coef_).mean(axis=0)
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'precision_per_class': precision_per_class,
            'recall_macro': recall_macro,
            'recall_per_class': recall_per_class,
            'f1_macro': f1_macro,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'feature_importance': feature_importance
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
        
    def _plot_feature_importance(self, results, stage_name, save_dir):
        """Create enhanced feature importance visualization for top features"""
        # Find models with feature importance
        models_with_importance = {}
        for model_name, result in results.items():
            if 'feature_importance' in result and result['feature_importance'] is not None:
                models_with_importance[model_name] = result['feature_importance']
        
        if not models_with_importance:
            logger.warning(f"No feature importance available for {stage_name} models")
            return
            
        # Get feature names
        feature_names = None
        if stage_name == 'text':
            X_text, _ = self.feature_loader.load_text_features("training")
            feature_names = X_text.columns.tolist()
        elif stage_name == 'financial':
            X_financial, _ = self.feature_loader.load_financial_features("training")
            feature_names = X_financial.columns.tolist()
        elif stage_name == 'ensemble':
            # For ensemble, feature names are the prediction vectors
            # We'll use a placeholder for now
            feature_names = [f"Feature_{i}" for i in range(len(list(models_with_importance.values())[0]))]
            
        if not feature_names:
            logger.warning(f"No feature names available for {stage_name} models")
            return
            
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Use the first model with feature importance
        model_name = list(models_with_importance.keys())[0]
        importance = models_with_importance[model_name]
        
        # Get top 20 features
        if len(feature_names) > 20:
            top_indices = np.argsort(importance)[-20:]
            top_importance = importance[top_indices]
            top_features = [feature_names[i] for i in top_indices]
        else:
            top_indices = np.argsort(importance)
            top_importance = importance[top_indices]
            top_features = [feature_names[i] for i in top_indices]
            
        # Plot horizontal bar chart
        plt.barh(range(len(top_features)), top_importance, color='skyblue')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'{stage_name.title()} - {model_name} - Top Features')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(save_dir / f'{stage_name}_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Saved {stage_name} feature importance plot to {save_dir}")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete 3-stage ML pipeline"""
        logger.info("🚀 Starting Complete 3-Stage ML Pipeline")
        
        # Stage 1: Text Models
        text_results = self.train_stage1_text_models()
        
        # Stage 2: Financial Models
        financial_results = self.train_stage2_financial_models()
        
        # Stage 3: Ensemble Models
        ensemble_results = self.train_stage3_ensemble_model(text_results, financial_results)
        
        # Generate feature importance plots
        self._plot_feature_importance(text_results['results'], 'text', self.text_model_dir)
        self._plot_feature_importance(financial_results['results'], 'financial', self.financial_model_dir)
        self._plot_feature_importance(ensemble_results['results'], 'ensemble', self.ensemble_model_dir)
        
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
            
            # Dataset overview
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 30 + "\n")
            try:
                X_text, y_text = self.feature_loader.load_text_features("training")
                X_financial, y_financial = self.feature_loader.load_financial_features("training")
                
                f.write(f"Text features: {X_text.shape[0]} samples, {X_text.shape[1]} features\n")
                f.write(f"Financial features: {X_financial.shape[0]} samples, {X_financial.shape[1]} features\n")
                
                # Class distribution
                class_counts = np.bincount(y_text)
                f.write("\nClass distribution:\n")
                for i, (class_name, count) in enumerate(zip(self.class_names, class_counts)):
                    pct = count / len(y_text) * 100
                    f.write(f"  {class_name}: {count} samples ({pct:.1f}%)\n")
                f.write("\n")
            except:
                f.write("Could not load dataset information\n\n")
            
            # Stage 1: Text Results
            f.write("STAGE 1: TEXT-BASED ML RESULTS\n")
            f.write("-" * 30 + "\n")
            for model_name, result in text_results['results'].items():
                f.write(f"{model_name.upper()}:\n")
                f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"  Precision (macro): {result['precision_macro']:.4f}\n")
                f.write(f"  Precision by class: Neg={result['precision_per_class'][0]:.3f}, Neu={result['precision_per_class'][1]:.3f}, Pos={result['precision_per_class'][2]:.3f}\n")
                f.write(f"  Recall by class: Neg={result['recall_per_class'][0]:.3f}, Neu={result['recall_per_class'][1]:.3f}, Pos={result['recall_per_class'][2]:.3f}\n")
                f.write(f"  F1 by class: Neg={result['f1_per_class'][0]:.3f}, Neu={result['f1_per_class'][1]:.3f}, Pos={result['f1_per_class'][2]:.3f}\n")
                f.write(f"  CV Score: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})\n\n")
                
            # Stage 2: Financial Results
            f.write("STAGE 2: FINANCIAL ML RESULTS\n")
            f.write("-" * 30 + "\n")
            for model_name, result in financial_results['results'].items():
                f.write(f"{model_name.upper()}:\n")
                f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"  Precision (macro): {result['precision_macro']:.4f}\n")
                f.write(f"  Precision by class: Neg={result['precision_per_class'][0]:.3f}, Neu={result['precision_per_class'][1]:.3f}, Pos={result['precision_per_class'][2]:.3f}\n")
                f.write(f"  Recall by class: Neg={result['recall_per_class'][0]:.3f}, Neu={result['recall_per_class'][1]:.3f}, Pos={result['recall_per_class'][2]:.3f}\n")
                f.write(f"  F1 by class: Neg={result['f1_per_class'][0]:.3f}, Neu={result['f1_per_class'][1]:.3f}, Pos={result['f1_per_class'][2]:.3f}\n")
                f.write(f"  CV Score: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})\n\n")
            
            # Domain Correlation Analysis
            f.write("DOMAIN CORRELATION ANALYSIS\n")
            f.write("-" * 30 + "\n")
            try:
                best_text = max(text_results['results'].items(), key=lambda x: x[1]['accuracy'])
                best_financial = max(financial_results['results'].items(), key=lambda x: x[1]['accuracy'])
                
                f.write(f"Prediction Agreement: {best_text[1]['prediction_agreement']:.3f} ({best_text[1]['prediction_agreement']*100:.1f}%)\n")
                f.write(f"Disagreement Rate: {best_text[1]['disagreement_rate']:.3f} ({best_text[1]['disagreement_rate']*100:.1f}%)\n")
                
                if 'confidence_correlation' in best_text[1]:
                    f.write(f"Confidence Correlation: {best_text[1]['confidence_correlation']:.3f}\n")
                
                f.write("\n")
            except:
                f.write("Could not generate correlation analysis\n\n")
                
            # Stage 3: Ensemble Results
            f.write("STAGE 3: ENSEMBLE ML RESULTS\n")
            f.write("-" * 30 + "\n")
            for model_name, result in ensemble_results['results'].items():
                f.write(f"{model_name.upper()}:\n")
                f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"  Precision (macro): {result['precision_macro']:.4f}\n")
                f.write(f"  Precision by class: Neg={result['precision_per_class'][0]:.3f}, Neu={result['precision_per_class'][1]:.3f}, Pos={result['precision_per_class'][2]:.3f}\n")
                f.write(f"  Recall by class: Neg={result['recall_per_class'][0]:.3f}, Neu={result['recall_per_class'][1]:.3f}, Pos={result['recall_per_class'][2]:.3f}\n")
                f.write(f"  F1 by class: Neg={result['f1_per_class'][0]:.3f}, Neu={result['f1_per_class'][1]:.3f}, Pos={result['f1_per_class'][2]:.3f}\n")
                f.write(f"  CV Score: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})\n\n")
                
            # Best models
            f.write("BEST PERFORMING MODELS\n")
            f.write("-" * 30 + "\n")
            
            # Find best models
            best_text = max(text_results['results'].items(), key=lambda x: x[1]['accuracy'])
            best_financial = max(financial_results['results'].items(), key=lambda x: x[1]['accuracy'])
            best_ensemble = max(ensemble_results['results'].items(), key=lambda x: x[1]['accuracy'])
            
            f.write(f"Best Text Model: {best_text[0]} (Accuracy: {best_text[1]['accuracy']:.4f})\n")
            f.write(f"Best Financial Model: {best_financial[0]} (Accuracy: {best_financial[1]['accuracy']:.4f})\n")
            f.write(f"Best Ensemble Model: {best_ensemble[0]} (Accuracy: {best_ensemble[1]['accuracy']:.4f})\n\n")
            
            # Performance comparison to baselines
            f.write("PERFORMANCE vs BASELINES\n")
            f.write("-" * 30 + "\n")
            random_accuracy = 1/3  # 33.3% for 3-class
            best_accuracy = best_ensemble[1]['accuracy']
            improvement = (best_accuracy - random_accuracy) / random_accuracy * 100
            
            f.write(f"Random baseline accuracy: {random_accuracy:.1%}\n")
            f.write(f"Best model accuracy: {best_accuracy:.1%}\n")
            f.write(f"Improvement over random: {improvement:.1f}%\n\n")
            
            # Trading interpretation
            f.write("TRADING INTERPRETATION\n")
            f.write("-" * 30 + "\n")
            best_precision_positive = best_ensemble[1]['precision_per_class'][2]
            f.write(f"- Positive class (BUY signal) precision of {best_precision_positive:.1%} means:\n")
            f.write(f"  When the model predicts a stock will be in the top 33.3% of performers,\n")
            f.write(f"  it is correct {best_precision_positive:.1%} of the time.\n")
            f.write(f"- This minimizes false positives (entering bad trades).\n\n")
            
            f.write("=" * 50 + "\n")
            
        logger.info(f"📄 Final report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='3-Stage ML Pipeline Training')
    parser.add_argument('--db-path', default=None, help='Path to DuckDB database (kept for backward compatibility)')
    parser.add_argument('--output-dir', default='models', help='Output directory for models')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--input-dir', default='data/ml_features/balanced', 
                       help='Directory containing balanced ML feature CSV files')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = ThreeStageMLPipeline(
        db_path=args.db_path,
        output_dir=args.output_dir,
        random_state=args.random_state
    )
    
    # Update feature loader with input directory
    pipeline.feature_loader = CorrectedCSVFeatureLoader(args.input_dir)
    
    try:
        results = pipeline.run_complete_pipeline()
        logger.info("🎉 Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        raise


if __name__ == '__main__':
    main() 