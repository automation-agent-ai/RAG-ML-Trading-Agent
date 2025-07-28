#!/usr/bin/env python3
"""
multiclass_text_model.py - Multi-Class Text-Only Model

Trains multi-class models using ONLY text-based features:
- News sentiment and events
- UserPosts community sentiment  
- Analyst recommendations sentiment

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

warnings.filterwarnings("ignore")

class MultiClassTextModel:
    """Multi-class model using ONLY text-based features"""
    
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
        
    def create_percentile_labels(self, outperformance_values):
        """Create 3-class labels based on percentiles"""
        p33 = self.label_thresholds['negative_threshold']
        p67 = self.label_thresholds['positive_threshold']
        
        labels = np.zeros(len(outperformance_values), dtype=int)
        labels[outperformance_values <= p33] = 0  # Negative
        labels[(outperformance_values > p33) & (outperformance_values <= p67)] = 1  # Neutral
        labels[outperformance_values > p67] = 2  # Positive
        
        return labels
    
    def load_text_features(self):
        """Load ONLY text-based features from all domains"""
        print("📰 Loading TEXT-ONLY features for multi-class prediction...")
        
        conn = duckdb.connect(self.db_path)
        
        # Load labels
        labels_query = """
        SELECT setup_id, outperformance_10d 
        FROM labels 
        WHERE outperformance_10d IS NOT NULL
        """
        labels_df = conn.execute(labels_query).df()
        
        # Create percentile-based labels
        outperformance_values = labels_df['outperformance_10d'].values
        multi_class_labels = self.create_percentile_labels(outperformance_values)
        labels_df['target'] = multi_class_labels
        
        print(f"📊 Label distribution:")
        for i, class_name in enumerate(self.class_names):
            count = (multi_class_labels == i).sum()
            pct = count / len(multi_class_labels) * 100
            print(f"   {class_name}: {count} ({pct:.1f}%)")
        
        # Load TEXT features ONLY
        features_data = []
        
        # News features (TEXT) - ALL AVAILABLE FEATURES
        try:
            news_query = """
            SELECT setup_id,
                   -- Sentiment and event features
                   sentiment_score_financial_results, sentiment_score_corporate_actions,
                   sentiment_score_governance, sentiment_score_corporate_events, sentiment_score_other_signals,
                   count_financial_results, count_corporate_actions, count_governance, count_corporate_events, count_other_signals,
                   max_severity_financial_results, max_severity_corporate_actions, max_severity_governance, 
                   max_severity_corporate_events, max_severity_other_signals,
                   avg_headline_spin_financial_results, avg_headline_spin_corporate_actions, avg_headline_spin_governance,
                   avg_headline_spin_corporate_events, avg_headline_spin_other_signals,
                   -- Event presence indicators
                   profit_warning_present, capital_raise_present, board_change_present, 
                   contract_award_present, merger_or_acquisition_present, broker_recommendation_present, credit_rating_change_present,
                   -- Synthetic summaries and explanations (rich text features)
                   CASE WHEN synthetic_summary_financial_results IS NOT NULL THEN 1 ELSE 0 END as has_financial_summary,
                   CASE WHEN synthetic_summary_corporate_actions IS NOT NULL THEN 1 ELSE 0 END as has_corporate_actions_summary,
                   CASE WHEN synthetic_summary_governance IS NOT NULL THEN 1 ELSE 0 END as has_governance_summary,
                   CASE WHEN synthetic_summary_corporate_events IS NOT NULL THEN 1 ELSE 0 END as has_corporate_events_summary,
                   CASE WHEN synthetic_summary_other_signals IS NOT NULL THEN 1 ELSE 0 END as has_other_signals_summary,
                   CASE WHEN cot_explanation_news_grouped IS NOT NULL THEN 1 ELSE 0 END as has_news_explanation
            FROM news_features
            """
            news_df = conn.execute(news_query).df()
            print(f"   ✅ News TEXT features (ALL): {len(news_df)} setups")
            features_data.append(('news', news_df))
        except Exception as e:
            print(f"   ⚠️ No news features: {e}")
        
        # UserPosts features (TEXT) - ALL AVAILABLE FEATURES
        try:
            posts_query = """
            SELECT setup_id,
                   -- Sentiment features
                   avg_sentiment, community_sentiment_score, bull_bear_ratio,
                   rumor_intensity, trusted_user_sentiment, relevance_score, engagement_score,
                   contrarian_signal, consensus_level, recent_sentiment_shift, coherence,
                   post_count, unique_users, sentiment_distribution,
                   -- Topic and discussion features
                   CASE WHEN consensus_topics IS NOT NULL THEN 1 ELSE 0 END as has_consensus_topics,
                   CASE WHEN controversial_topics IS NOT NULL THEN 1 ELSE 0 END as has_controversial_topics,
                   CASE WHEN synthetic_post IS NOT NULL THEN 1 ELSE 0 END as has_synthetic_post,
                   CASE WHEN cot_explanation IS NOT NULL THEN 1 ELSE 0 END as has_userposts_explanation
            FROM userposts_features
            """
            posts_df = conn.execute(posts_query).df()
            print(f"   ✅ UserPosts TEXT features (ALL): {len(posts_df)} setups")
            features_data.append(('userposts', posts_df))
        except Exception as e:
            print(f"   ⚠️ No userposts features: {e}")
        
        # Analyst features (TEXT) - ALL AVAILABLE FEATURES
        try:
            analyst_query = """
            SELECT setup_id,
                   -- Recommendation features
                   recommendation_count, buy_recommendations, sell_recommendations, hold_recommendations,
                   avg_price_target, price_target_vs_current, price_target_spread,
                   coverage_breadth, consensus_rating, recent_upgrades, recent_downgrades,
                   analyst_conviction_score, recommendation_momentum,
                   -- Synthetic summaries and explanations
                   CASE WHEN synthetic_analyst_summary IS NOT NULL THEN 1 ELSE 0 END as has_analyst_summary,
                   CASE WHEN cot_explanation_analyst IS NOT NULL THEN 1 ELSE 0 END as has_analyst_explanation
            FROM analyst_recommendations_features
            """
            analyst_df = conn.execute(analyst_query).df()
            print(f"   ✅ Analyst TEXT features (ALL): {len(analyst_df)} setups")
            features_data.append(('analyst', analyst_df))
        except Exception as e:
            print(f"   ⚠️ No analyst features: {e}")
        
        conn.close()
        
        # Merge all TEXT features
        if not features_data:
            raise ValueError("No text feature data found!")
        
        # Start with first domain
        domain_name, merged_df = features_data[0]
        print(f"   Starting with {domain_name}: {len(merged_df)} setups")
        
        # Merge other domains
        for domain_name, domain_df in features_data[1:]:
            before_count = len(merged_df)
            merged_df = merged_df.merge(domain_df, on='setup_id', how='outer', suffixes=('', f'_{domain_name}'))
            after_count = len(merged_df)
            print(f"   Merged {domain_name}: {before_count} -> {after_count} setups")
        
        # Merge with labels
        final_df = merged_df.merge(labels_df[['setup_id', 'target']], on='setup_id', how='inner')
        print(f"   📊 Final TEXT dataset: {len(final_df)} setups with labels")
        
        # Prepare features
        feature_cols = [col for col in final_df.columns if col not in ['setup_id', 'target']]
        X = final_df[feature_cols].copy()
        y = final_df['target']
        setup_ids = final_df['setup_id']
        
        # Clean features
        X_clean = self._preprocess_features(X)
        
        print(f"   📊 Processed TEXT features: {X_clean.shape[1]} columns, {X_clean.shape[0]} samples")
        print(f"   📊 Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X_clean, y, setup_ids
    
    def _preprocess_features(self, X):
        """Clean feature preprocessing for text features"""
        # Filter numeric columns
        numeric_cols = []
        for col in X.columns:
            if X[col].dtype in ['object', 'string']:
                try:
                    X.loc[:, col] = pd.to_numeric(X[col], errors='coerce')
                    numeric_cols.append(col)
                except:
                    continue
            elif 'datetime' in str(X[col].dtype).lower():
                continue
            elif X[col].dtype in ['bool']:
                X.loc[:, col] = X[col].astype(int)
                numeric_cols.append(col)
            else:
                numeric_cols.append(col)
        
        # Keep only numeric columns and remove all-NaN columns
        X_clean = X[numeric_cols].dropna(axis=1, how='all')
        
        # Store feature names
        self.feature_names = X_clean.columns.tolist()
        print(f"   📊 TEXT feature names: {len(self.feature_names)} features")
        print(f"   📊 Sample features: {self.feature_names[:5]}")
        
        # Impute missing values
        X_imputed = self.imputer.fit_transform(X_clean)
        X_final = pd.DataFrame(X_imputed, columns=self.feature_names, index=X_clean.index)
        
        return X_final
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train text-based multi-class models"""
        print("🚀 Training TEXT-ONLY multi-class models...")
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        print(f"   📊 Class weights: {class_weight_dict}")
        
        # Define models optimized for text features
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
            print(f"   🎯 Training {name} on TEXT features...")
            
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
        
        model_path = self.save_dir / 'text_multiclass_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': best_model,
                'scaler': self.scaler if best_model_name == 'LogisticRegression' else None,
                'imputer': self.imputer,
                'feature_names': self.feature_names,
                'model_name': best_model_name,
                'thresholds': self.label_thresholds
            }, f)
        
        print(f"   💾 Best TEXT model ({best_model_name}) saved to {model_path}")
        
        self.models = {name: results[name]['model'] for name in results}
        return results
    
    def cross_validate_models(self, X, y, cv_folds=5):
        """Cross-validation for text models"""
        print(f"🔍 Cross-validating TEXT models ({cv_folds}-fold)...")
        
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
    
    def generate_predictions(self, X):
        """Generate predictions using trained text model"""
        if not self.models:
            raise ValueError("No trained models available. Run train_models first.")
        
        # Use best model (highest positive class precision)
        best_model_name = max(self.models.keys(), 
                             key=lambda k: getattr(self.models[k], '_positive_precision', 0))
        best_model = self.models[best_model_name]
        
        # Scale if needed
        if best_model_name == 'LogisticRegression':
            X_scaled = self.scaler.transform(X)
            predictions = best_model.predict(X_scaled)
            probabilities = best_model.predict_proba(X_scaled)
        else:
            predictions = best_model.predict(X)
            probabilities = best_model.predict_proba(X)
        
        return predictions, probabilities, best_model_name

def main():
    """Run text-only multi-class analysis"""
    print("🎯 Multi-Class TEXT-ONLY Model Training")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = MultiClassTextModel()
    
    # Load text features only
    X, y, setup_ids = analyzer.load_text_features()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Data split: {len(X_train)} train, {len(X_test)} test")
    
    # Train models
    train_results = analyzer.train_models(X_train, y_train, X_test, y_test)
    
    # Cross-validation
    cv_results = analyzer.cross_validate_models(X, y)
    
    print(f"\n🎉 TEXT-ONLY multi-class analysis complete!")
    print(f"📁 Results saved in: {analyzer.save_dir}/")
    
    return analyzer, train_results, cv_results

if __name__ == "__main__":
    analyzer, train_results, cv_results = main() 