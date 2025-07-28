#!/usr/bin/env python3
"""
multiclass_ensemble_model.py - Multi-Class Ensemble Model

Trains multi-class ensemble models using predictions from:
- Text-only model (news + userposts + analyst sentiment)
- Fundamentals-only model (financial ratios + metrics)

This is a meta-learner that combines the prediction vectors from the two specialized models.
Target: 3-class (Negative ≤-1.93%, Neutral, Positive >2.47%)
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import duckdb
import pickle

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, precision_recall_curve, roc_curve,
    auc, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Import the specialized models
from ml.multiclass_text_model import MultiClassTextModel
from ml.multiclass_fundamentals_model import MultiClassFundamentalsModel

warnings.filterwarnings("ignore")

class MultiClassEnsembleModel:
    """Ensemble multi-class model using predictions from text and fundamentals models"""
    
    def __init__(self, db_path="data/sentiment_system.duckdb", save_dir="analysis_ml_multiclass"):
        self.db_path = db_path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.cv_results = {}
        self.feature_names = []
        self.label_thresholds = {
            'negative_threshold': -1.9273,  # 33rd percentile 
            'positive_threshold': 2.4740   # 67th percentile
        }
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
        # Class information
        self.class_names = ['Negative', 'Neutral', 'Positive']
        self.class_colors = ['#d62728', '#ff7f0e', '#2ca02c']  # Red, Orange, Green
        
        # Component models
        self.text_model = None
        self.fundamentals_model = None
        
    def create_percentile_labels(self, outperformance_values):
        """Create 3-class labels based on percentiles"""
        p33 = self.label_thresholds['negative_threshold']
        p67 = self.label_thresholds['positive_threshold']
        
        labels = np.zeros(len(outperformance_values), dtype=int)
        labels[outperformance_values <= p33] = 0  # Negative
        labels[(outperformance_values > p33) & (outperformance_values <= p67)] = 1  # Neutral
        labels[outperformance_values > p67] = 2  # Positive
        
        return labels
    
    def train_component_models(self):
        """Train the text and fundamentals component models"""
        print("🎯 Training component models for ensemble...")
        
        # Train text model
        print("\n📰 Training TEXT component model...")
        self.text_model = MultiClassTextModel(self.db_path, self.save_dir)
        X_text, y_text, setup_ids_text = self.text_model.load_text_features()
        
        X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
            X_text, y_text, test_size=0.2, random_state=42, stratify=y_text
        )
        
        text_results = self.text_model.train_models(X_train_text, y_train_text, X_test_text, y_test_text)
        text_cv_results = self.text_model.cross_validate_models(X_text, y_text)
        
        # Train fundamentals model
        print("\n💼 Training FUNDAMENTALS component model...")
        self.fundamentals_model = MultiClassFundamentalsModel(self.db_path, self.save_dir)
        X_fund, y_fund, setup_ids_fund = self.fundamentals_model.load_fundamentals_features()
        
        X_train_fund, X_test_fund, y_train_fund, y_test_fund = train_test_split(
            X_fund, y_fund, test_size=0.2, random_state=42, stratify=y_fund
        )
        
        fund_results = self.fundamentals_model.train_models(X_train_fund, y_train_fund, X_test_fund, y_test_fund)
        fund_cv_results = self.fundamentals_model.cross_validate_models(X_fund, y_fund)
        
        print(f"\n📊 Component model training complete!")
        return text_results, fund_results, text_cv_results, fund_cv_results
    
    def load_ensemble_features(self):
        """Load features by generating predictions from both component models"""
        print("🏆 Loading ENSEMBLE features from component model predictions...")
        
        # Get overlapping setup_ids from both models
        text_X, text_y, text_setup_ids = self.text_model.load_text_features()
        fund_X, fund_y, fund_setup_ids = self.fundamentals_model.load_fundamentals_features()
        
        # Find common setup_ids
        common_setup_ids = set(text_setup_ids) & set(fund_setup_ids)
        common_setup_ids = list(common_setup_ids)
        
        print(f"   📊 Common setup_ids: {len(common_setup_ids)}")
        print(f"   📊 Text-only: {len(text_setup_ids)}, Fundamentals-only: {len(fund_setup_ids)}")
        
        # Filter data to common setup_ids
        text_mask = text_setup_ids.isin(common_setup_ids)
        fund_mask = fund_setup_ids.isin(common_setup_ids)
        
        text_X_common = text_X[text_mask]
        text_y_common = text_y[text_mask]
        text_setup_ids_common = text_setup_ids[text_mask]
        
        fund_X_common = fund_X[fund_mask]
        fund_y_common = fund_y[fund_mask]
        fund_setup_ids_common = fund_setup_ids[fund_mask]
        
        # Align by setup_id (sort both by setup_id)
        text_df = pd.DataFrame({'setup_id': text_setup_ids_common, 'y': text_y_common})
        text_df = text_df.merge(text_X_common, left_index=True, right_index=True)
        text_df = text_df.sort_values('setup_id')
        
        fund_df = pd.DataFrame({'setup_id': fund_setup_ids_common, 'y': fund_y_common})
        fund_df = fund_df.merge(fund_X_common, left_index=True, right_index=True)
        fund_df = fund_df.sort_values('setup_id')
        
        # Verify alignment
        assert list(text_df['setup_id']) == list(fund_df['setup_id']), "Setup IDs not aligned!"
        assert list(text_df['y']) == list(fund_df['y']), "Labels not aligned!"
        
        # Generate predictions from component models
        print("   🎯 Generating text model predictions...")
        text_predictions, text_probabilities, text_model_name = self.text_model.generate_predictions(
            text_df.drop(['setup_id', 'y'], axis=1)
        )
        
        print("   🎯 Generating fundamentals model predictions...")
        fund_predictions, fund_probabilities, fund_model_name = self.fundamentals_model.generate_predictions(
            fund_df.drop(['setup_id', 'y'], axis=1)
        )
        
        # Analyze prediction correlation
        self.analyze_prediction_correlation(text_predictions, fund_predictions, text_probabilities, fund_probabilities)
        
        # Create ensemble features
        ensemble_features = []
        
        # Add class predictions
        ensemble_features.extend([
            ('text_pred_negative', (text_predictions == 0).astype(int)),
            ('text_pred_neutral', (text_predictions == 1).astype(int)),
            ('text_pred_positive', (text_predictions == 2).astype(int)),
            ('fund_pred_negative', (fund_predictions == 0).astype(int)),
            ('fund_pred_neutral', (fund_predictions == 1).astype(int)),
            ('fund_pred_positive', (fund_predictions == 2).astype(int))
        ])
        
        # Add class probabilities
        for i, class_name in enumerate(['negative', 'neutral', 'positive']):
            ensemble_features.extend([
                (f'text_prob_{class_name}', text_probabilities[:, i]),
                (f'fund_prob_{class_name}', fund_probabilities[:, i])
            ])
        
        # Add confidence measures
        text_confidence = np.max(text_probabilities, axis=1)
        fund_confidence = np.max(fund_probabilities, axis=1)
        ensemble_features.extend([
            ('text_confidence', text_confidence),
            ('fund_confidence', fund_confidence),
            ('confidence_diff', np.abs(text_confidence - fund_confidence)),
            ('confidence_avg', (text_confidence + fund_confidence) / 2)
        ])
        
        # Add agreement measures
        prediction_agreement = (text_predictions == fund_predictions).astype(int)
        ensemble_features.append(('prediction_agreement', prediction_agreement))
        
        # Create DataFrame
        X_ensemble = pd.DataFrame({name: values for name, values in ensemble_features})
        y_ensemble = text_df['y'].values  # Same as fund_df['y']
        setup_ids_ensemble = text_df['setup_id'].values
        
        self.feature_names = X_ensemble.columns.tolist()
        
        print(f"   📊 Ensemble features: {X_ensemble.shape[1]} columns, {X_ensemble.shape[0]} samples")
        print(f"   📊 Feature names: {self.feature_names}")
        print(f"   📊 Class distribution: {dict(zip(*np.unique(y_ensemble, return_counts=True)))}")
        
        return X_ensemble, y_ensemble, setup_ids_ensemble
    
    def analyze_prediction_correlation(self, text_preds, fund_preds, text_probs, fund_probs):
        """Analyze correlation between text and fundamentals predictions"""
        print("\n🔍 PREDICTION CORRELATION ANALYSIS")
        print("=" * 50)
        
        # Prediction agreement
        agreement = (text_preds == fund_preds).mean()
        print(f"📊 Prediction Agreement: {agreement:.3f} ({agreement*100:.1f}%)")
        
        # Class-wise agreement
        for i, class_name in enumerate(self.class_names):
            text_class_mask = (text_preds == i)
            fund_class_mask = (fund_preds == i)
            
            if text_class_mask.sum() > 0 and fund_class_mask.sum() > 0:
                class_agreement = (text_class_mask & fund_class_mask).sum() / max(text_class_mask.sum(), fund_class_mask.sum())
                print(f"   {class_name} class agreement: {class_agreement:.3f}")
        
        # Probability correlations
        print(f"\n📊 Probability Correlations:")
        for i, class_name in enumerate(self.class_names):
            corr, p_value = pearsonr(text_probs[:, i], fund_probs[:, i])
            print(f"   {class_name} probabilities: r={corr:.3f} (p={p_value:.3f})")
        
        # Overall correlation of max probabilities (confidence)
        text_confidence = np.max(text_probs, axis=1)
        fund_confidence = np.max(fund_probs, axis=1)
        confidence_corr, confidence_p = pearsonr(text_confidence, fund_confidence)
        print(f"   Confidence correlation: r={confidence_corr:.3f} (p={confidence_p:.3f})")
        
        # Disagreement analysis
        disagreement_mask = (text_preds != fund_preds)
        disagreement_rate = disagreement_mask.mean()
        print(f"\n🤔 Disagreement Analysis:")
        print(f"   Disagreement rate: {disagreement_rate:.3f} ({disagreement_rate*100:.1f}%)")
        
        if disagreement_rate > 0:
            # Where text says positive but fundamentals says negative (and vice versa)
            text_pos_fund_neg = ((text_preds == 2) & (fund_preds == 0)).sum()
            text_neg_fund_pos = ((text_preds == 0) & (fund_preds == 2)).sum()
            print(f"   Text Positive, Fund Negative: {text_pos_fund_neg}")
            print(f"   Text Negative, Fund Positive: {text_neg_fund_pos}")
        
        # Save correlation analysis
        correlation_results = {
            'prediction_agreement': agreement,
            'probability_correlations': {
                class_name: pearsonr(text_probs[:, i], fund_probs[:, i])[0] 
                for i, class_name in enumerate(self.class_names)
            },
            'confidence_correlation': confidence_corr,
            'disagreement_rate': disagreement_rate
        }
        
        with open(self.save_dir / 'prediction_correlation_analysis.pkl', 'wb') as f:
            pickle.dump(correlation_results, f)
        
        print(f"   💾 Correlation analysis saved to {self.save_dir}/prediction_correlation_analysis.pkl")
        
        return correlation_results
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train ensemble multi-class models"""
        print("🚀 Training ENSEMBLE multi-class models...")
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        print(f"   📊 Class weights: {class_weight_dict}")
        
        # Define models optimized for ensemble features
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'LogisticRegression': LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
        }
        
        # Scale features for LogisticRegression
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, model in models.items():
            print(f"   🎯 Training {name} on ENSEMBLE features...")
            
            # Use scaled features for LogisticRegression
            if name == 'LogisticRegression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
            
            # Calculate comprehensive metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
            precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            # Class-wise precision
            precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # For logistic regression, use absolute coefficients
                feature_importance = np.abs(model.coef_).mean(axis=0)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'precision_weighted': precision_weighted,
                'precision_per_class': precision_per_class,
                'recall_macro': recall_macro,
                'recall_per_class': recall_per_class,
                'f1_macro': f1_macro,
                'confusion_matrix': cm,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'feature_importance': feature_importance
            }
            
            print(f"     📊 Accuracy: {accuracy:.3f}")
            print(f"     📊 Precision (macro): {precision_macro:.3f}")
            print(f"     📊 Precision (Positive class): {precision_per_class[2]:.3f}")
        
        # Save best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['precision_per_class'][2])
        best_model = results[best_model_name]['model']
        
        model_path = self.save_dir / 'ensemble_multiclass_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': best_model,
                'scaler': self.scaler if best_model_name == 'LogisticRegression' else None,
                'imputer': self.imputer,
                'feature_names': self.feature_names,
                'model_name': best_model_name,
                'thresholds': self.label_thresholds
            }, f)
        
        print(f"   💾 Best ENSEMBLE model ({best_model_name}) saved to {model_path}")
        
        self.models = {name: results[name]['model'] for name in results}
        return results
    
    def cross_validate_models(self, X, y, cv_folds=5):
        """Cross-validation for ensemble models"""
        print(f"🔍 Cross-validating ENSEMBLE models ({cv_folds}-fold)...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = {}
        
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss'),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42, verbose=-1),
            'LogisticRegression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        }
        
        for name, model in models.items():
            print(f"   📊 Cross-validating {name}...")
            
            fold_results = {
                'accuracy': [],
                'precision_macro': [],
                'precision_positive': [],
                'recall_macro': [],
                'f1_macro': []
            }
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                # Scale for LogisticRegression
                if name == 'LogisticRegression':
                    scaler = StandardScaler()
                    X_train_fold = scaler.fit_transform(X_train_fold)
                    X_val_fold = scaler.transform(X_val_fold)
                
                # Train and predict
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                
                # Calculate metrics
                fold_results['accuracy'].append(accuracy_score(y_val_fold, y_pred))
                fold_results['precision_macro'].append(precision_score(y_val_fold, y_pred, average='macro', zero_division=0))
                fold_results['precision_positive'].append(precision_score(y_val_fold, y_pred, labels=[2], average='macro', zero_division=0))
                fold_results['recall_macro'].append(recall_score(y_val_fold, y_pred, average='macro', zero_division=0))
                fold_results['f1_macro'].append(f1_score(y_val_fold, y_pred, average='macro', zero_division=0))
            
            # Calculate means and stds
            cv_results[name] = {
                metric: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
                for metric, values in fold_results.items()
            }
            
            print(f"     📊 Accuracy: {cv_results[name]['accuracy']['mean']:.3f} ± {cv_results[name]['accuracy']['std']:.3f}")
            print(f"     📊 Precision (Positive): {cv_results[name]['precision_positive']['mean']:.3f} ± {cv_results[name]['precision_positive']['std']:.3f}")
        
        self.cv_results = cv_results
        return cv_results

def main():
    """Run ensemble multi-class analysis"""
    print("🎯 Multi-Class ENSEMBLE Model Training")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = MultiClassEnsembleModel()
    
    # Train component models first
    text_results, fund_results, text_cv, fund_cv = analyzer.train_component_models()
    
    # Load ensemble features (using component model predictions)
    X, y, setup_ids = analyzer.load_ensemble_features()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Data split: {len(X_train)} train, {len(X_test)} test")
    
    # Train ensemble models
    train_results = analyzer.train_models(X_train, y_train, X_test, y_test)
    
    # Cross-validation
    cv_results = analyzer.cross_validate_models(X, y)
    
    print(f"\n🎉 ENSEMBLE multi-class analysis complete!")
    print(f"📁 Results saved in: {analyzer.save_dir}/")
    
    return analyzer, train_results, cv_results

if __name__ == "__main__":
    analyzer, train_results, cv_results = main() 