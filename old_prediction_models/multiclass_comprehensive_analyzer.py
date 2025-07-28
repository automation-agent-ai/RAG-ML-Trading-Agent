#!/usr/bin/env python3
"""
multiclass_comprehensive_analyzer.py - Comprehensive Multi-Class Analysis

Complete analysis of 3-class percentile-based classification:
- Negative: Bottom 33.3% outperformance (≤-1.93%)
- Neutral: Middle 33.3% outperformance (-1.93% to 2.47%)
- Positive: Top 33.3% outperformance (>2.47%)

Generates detailed reports and visualizations similar to other ML analysis folders.
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
from matplotlib.colors import ListedColormap

warnings.filterwarnings("ignore")

class MultiClassComprehensiveAnalyzer:
    """Comprehensive analysis for multi-class stock outperformance prediction"""
    
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
    
    def load_and_prepare_data(self):
        """Load all features and create multi-class labels"""
        print("📊 Loading and preparing multi-class dataset...")
        
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
            threshold = "≤" + str(self.label_thresholds['negative_threshold']) if i == 0 else \
                       f"{self.label_thresholds['negative_threshold']} to {self.label_thresholds['positive_threshold']}" if i == 1 else \
                       ">" + str(self.label_thresholds['positive_threshold'])
            print(f"   {class_name} ({threshold}%): {count} ({pct:.1f}%)")
        
        # Load features from all domains
        features_data = []
        
        # Fundamentals features
        try:
            fund_df = conn.execute("SELECT * FROM fundamentals_features").df()
            print(f"   ✅ Fundamentals features: {len(fund_df)} setups")
            features_data.append(('fundamentals', fund_df))
        except Exception as e:
            print(f"   ⚠️ No fundamentals features: {e}")
        
        # News features  
        try:
            news_df = conn.execute("SELECT * FROM news_features").df()
            print(f"   ✅ News features: {len(news_df)} setups")
            features_data.append(('news', news_df))
        except Exception as e:
            print(f"   ⚠️ No news features: {e}")
        
        # UserPosts features
        try:
            posts_df = conn.execute("SELECT * FROM userposts_features").df()
            print(f"   ✅ UserPosts features: {len(posts_df)} setups")
            features_data.append(('userposts', posts_df))
        except Exception as e:
            print(f"   ⚠️ No userposts features: {e}")
        
        # Analyst features
        try:
            analyst_df = conn.execute("SELECT * FROM analyst_recommendations_features").df()
            print(f"   ✅ Analyst features: {len(analyst_df)} setups")
            features_data.append(('analyst', analyst_df))
        except Exception as e:
            print(f"   ⚠️ No analyst features: {e}")
        
        conn.close()
        
        # Merge all features
        if not features_data:
            raise ValueError("No feature data found!")
        
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
        print(f"   📊 Final dataset: {len(final_df)} setups with labels")
        
        # Prepare features
        feature_cols = [col for col in final_df.columns if col not in ['setup_id', 'target']]
        X = final_df[feature_cols].copy()
        y = final_df['target']
        setup_ids = final_df['setup_id']
        
        # Clean features
        X_clean = self._preprocess_features(X)
        
        print(f"   📊 Processed features: {X_clean.shape[1]} columns, {X_clean.shape[0]} samples")
        print(f"   📊 Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X_clean, y, setup_ids
    
    def _preprocess_features(self, X):
        """Clean feature preprocessing"""
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
        
        # Impute missing values
        X_imputed = self.imputer.fit_transform(X_clean)
        X_final = pd.DataFrame(X_imputed, columns=self.feature_names, index=X_clean.index)
        
        return X_final
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models with class balancing"""
        print("🚀 Training multi-class models with comprehensive evaluation...")
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        print(f"   📊 Class weights: {class_weight_dict}")
        
        # Define models
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
            print(f"   🎯 Training {name}...")
            
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
            print(f"     📊 Recall (macro): {recall_macro:.3f}")
            print(f"     📊 F1 (macro): {f1_macro:.3f}")
        
        self.models = {name: results[name]['model'] for name in results}
        return results
    
    def cross_validate_models(self, X, y, cv_folds=5):
        """Comprehensive cross-validation"""
        print(f"🔍 Performing {cv_folds}-fold stratified cross-validation...")
        
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
    
    def create_visualizations(self, train_results, cv_results, X_test, y_test):
        """Create comprehensive visualizations"""
        print("📊 Creating comprehensive visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model Performance Comparison
        self._plot_model_comparison(train_results, cv_results)
        
        # 2. Feature Importance Analysis
        self._plot_feature_importance(train_results)
        
        # 3. Confusion Matrices
        self._plot_confusion_matrices(train_results)
        
        # 4. Class-wise Performance
        self._plot_classwise_performance(train_results)
        
        # 5. Cross-validation Results
        self._plot_cv_results(cv_results)
        
        # 6. Feature Correlation Heatmap
        self._plot_feature_correlation(X_test)
        
        print(f"📊 All visualizations saved to {self.save_dir}/")
    
    def _plot_model_comparison(self, results, cv_results):
        """Plot model performance comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(results.keys())
        
        # Accuracy comparison
        test_acc = [results[model]['accuracy'] for model in models]
        cv_acc = [cv_results[model]['accuracy']['mean'] for model in models]
        cv_std = [cv_results[model]['accuracy']['std'] for model in models]
        
        x = np.arange(len(models))
        axes[0,0].bar(x - 0.2, test_acc, 0.4, label='Test Set', color='skyblue')
        axes[0,0].errorbar(x + 0.2, cv_acc, yerr=cv_std, fmt='o', label='CV Mean ± Std', color='orange')
        axes[0,0].set_title('Model Accuracy Comparison')
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(models, rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(alpha=0.3)
        
        # Positive class precision
        test_prec_pos = [results[model]['precision_per_class'][2] for model in models]
        cv_prec_pos = [cv_results[model]['precision_positive']['mean'] for model in models]
        cv_prec_std = [cv_results[model]['precision_positive']['std'] for model in models]
        
        axes[0,1].bar(x - 0.2, test_prec_pos, 0.4, label='Test Set', color='lightgreen')
        axes[0,1].errorbar(x + 0.2, cv_prec_pos, yerr=cv_prec_std, fmt='o', label='CV Mean ± Std', color='darkgreen')
        axes[0,1].set_title('Positive Class Precision (Most Important)')
        axes[0,1].set_xlabel('Models')
        axes[0,1].set_ylabel('Precision')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(models, rotation=45)
        axes[0,1].legend()
        axes[0,1].grid(alpha=0.3)
        
        # F1 Score comparison
        test_f1 = [results[model]['f1_macro'] for model in models]
        cv_f1 = [cv_results[model]['f1_macro']['mean'] for model in models]
        cv_f1_std = [cv_results[model]['f1_macro']['std'] for model in models]
        
        axes[1,0].bar(x - 0.2, test_f1, 0.4, label='Test Set', color='lightcoral')
        axes[1,0].errorbar(x + 0.2, cv_f1, yerr=cv_f1_std, fmt='o', label='CV Mean ± Std', color='darkred')
        axes[1,0].set_title('F1 Score (Macro Average)')
        axes[1,0].set_xlabel('Models')
        axes[1,0].set_ylabel('F1 Score')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(models, rotation=45)
        axes[1,0].legend()
        axes[1,0].grid(alpha=0.3)
        
        # Precision-Recall by class
        for i, class_name in enumerate(self.class_names):
            class_precision = [results[model]['precision_per_class'][i] for model in models]
            axes[1,1].bar(x + i*0.25, class_precision, 0.25, label=f'{class_name} Precision', 
                         color=self.class_colors[i], alpha=0.7)
        
        axes[1,1].set_title('Precision by Class (RandomForest)')
        axes[1,1].set_xlabel('Models')
        axes[1,1].set_ylabel('Precision')
        axes[1,1].set_xticks(x + 0.25)
        axes[1,1].set_xticklabels(models, rotation=45)
        axes[1,1].legend()
        axes[1,1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'multiclass_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, results):
        """Plot feature importance for models that support it"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.flatten()
        
        plot_idx = 0
        for name, result in results.items():
            if result['feature_importance'] is not None and plot_idx < 4:
                importance = result['feature_importance']
                feature_names = self.feature_names
                
                # Get top 15 features
                top_indices = np.argsort(importance)[-15:]
                top_importance = importance[top_indices]
                top_features = [feature_names[i] for i in top_indices]
                
                axes[plot_idx].barh(range(len(top_features)), top_importance, color='skyblue')
                axes[plot_idx].set_yticks(range(len(top_features)))
                axes[plot_idx].set_yticklabels(top_features)
                axes[plot_idx].set_xlabel('Feature Importance')
                axes[plot_idx].set_title(f'{name} - Top 15 Features')
                axes[plot_idx].grid(alpha=0.3)
                
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'multiclass_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, results):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (name, result) in enumerate(results.items()):
            cm = result['confusion_matrix']
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names,
                       ax=axes[idx])
            axes[idx].set_title(f'{name} - Confusion Matrix (Normalized)')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'multiclass_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_classwise_performance(self, results):
        """Plot class-wise precision and recall"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        models = list(results.keys())
        x = np.arange(len(models))
        width = 0.25
        
        # Precision by class
        for i, class_name in enumerate(self.class_names):
            class_precision = [results[model]['precision_per_class'][i] for model in models]
            axes[0].bar(x + i*width, class_precision, width, label=f'{class_name}', 
                       color=self.class_colors[i], alpha=0.8)
        
        axes[0].set_title('Precision by Class')
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('Precision')
        axes[0].set_xticks(x + width)
        axes[0].set_xticklabels(models, rotation=45)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Recall by class
        for i, class_name in enumerate(self.class_names):
            class_recall = [results[model]['recall_per_class'][i] for model in models]
            axes[1].bar(x + i*width, class_recall, width, label=f'{class_name}', 
                       color=self.class_colors[i], alpha=0.8)
        
        axes[1].set_title('Recall by Class')
        axes[1].set_xlabel('Models')
        axes[1].set_ylabel('Recall')
        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels(models, rotation=45)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'multiclass_classwise_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cv_results(self, cv_results):
        """Plot cross-validation results with error bars"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(cv_results.keys())
        metrics = ['accuracy', 'precision_macro', 'precision_positive', 'f1_macro']
        metric_titles = ['Accuracy', 'Precision (Macro)', 'Precision (Positive)', 'F1 Score (Macro)']
        
        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[idx // 2, idx % 2]
            
            means = [cv_results[model][metric]['mean'] for model in models]
            stds = [cv_results[model][metric]['std'] for model in models]
            
            x = np.arange(len(models))
            ax.bar(x, means, yerr=stds, capsize=5, color='lightblue', edgecolor='darkblue')
            ax.set_title(f'Cross-Validation: {title}')
            ax.set_xlabel('Models')
            ax.set_ylabel(title)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45)
            ax.grid(alpha=0.3)
            
            # Add value labels on bars
            for i, (mean, std) in enumerate(zip(means, stds)):
                ax.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'multiclass_cv_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_correlation(self, X):
        """Plot feature correlation heatmap"""
        # Calculate correlation matrix for top features
        if len(self.feature_names) > 20:
            # Use top 20 features from RandomForest if available
            if 'RandomForest' in self.models:
                rf_importance = self.models['RandomForest'].feature_importances_
                top_indices = np.argsort(rf_importance)[-20:]
                top_features = [self.feature_names[i] for i in top_indices]
                X_subset = X[top_features]
            else:
                X_subset = X[self.feature_names[:20]]
        else:
            X_subset = X[self.feature_names]
        
        corr_matrix = X_subset.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=ax, fmt='.2f')
        ax.set_title('Feature Correlation Matrix (Top Features)')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'multiclass_feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, train_results, cv_results, X, y):
        """Generate comprehensive text report"""
        print("📝 Generating comprehensive analysis report...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MULTI-CLASS OUTPERFORMANCE PREDICTION - COMPREHENSIVE ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Dataset overview
        report_lines.append("DATASET OVERVIEW")
        report_lines.append("-" * 40)
        report_lines.append(f"Total samples: {len(y)}")
        report_lines.append(f"Features: {len(self.feature_names)}")
        report_lines.append("")
        
        # Class distribution
        report_lines.append("CLASS DISTRIBUTION (Percentile-based)")
        report_lines.append("-" * 40)
        class_counts = np.bincount(y)
        for i, (class_name, count) in enumerate(zip(self.class_names, class_counts)):
            pct = count / len(y) * 100
            if i == 0:
                threshold = f"≤{self.label_thresholds['negative_threshold']:.2f}%"
            elif i == 1:
                threshold = f"{self.label_thresholds['negative_threshold']:.2f}% to {self.label_thresholds['positive_threshold']:.2f}%"
            else:
                threshold = f">{self.label_thresholds['positive_threshold']:.2f}%"
            report_lines.append(f"{class_name} ({threshold}): {count} samples ({pct:.1f}%)")
        report_lines.append("")
        
        # Model performance summary
        report_lines.append("MODEL PERFORMANCE SUMMARY")
        report_lines.append("-" * 40)
        for name, result in train_results.items():
            report_lines.append(f"\n{name}:")
            report_lines.append(f"  Accuracy: {result['accuracy']:.3f}")
            report_lines.append(f"  Precision (macro): {result['precision_macro']:.3f}")
            report_lines.append(f"  Precision (Positive class): {result['precision_per_class'][2]:.3f}")
            report_lines.append(f"  Recall (macro): {result['recall_macro']:.3f}")
            report_lines.append(f"  F1 Score (macro): {result['f1_macro']:.3f}")
        
        # Cross-validation results
        report_lines.append("\n\nCROSS-VALIDATION RESULTS (5-Fold)")
        report_lines.append("-" * 40)
        for name, cv_result in cv_results.items():
            report_lines.append(f"\n{name}:")
            report_lines.append(f"  Accuracy: {cv_result['accuracy']['mean']:.3f} ± {cv_result['accuracy']['std']:.3f}")
            report_lines.append(f"  Precision (Positive): {cv_result['precision_positive']['mean']:.3f} ± {cv_result['precision_positive']['std']:.3f}")
            report_lines.append(f"  F1 Score: {cv_result['f1_macro']['mean']:.3f} ± {cv_result['f1_macro']['std']:.3f}")
        
        # Feature importance
        report_lines.append("\n\nTOP FEATURES (RandomForest)")
        report_lines.append("-" * 40)
        if 'RandomForest' in train_results and train_results['RandomForest']['feature_importance'] is not None:
            importance = train_results['RandomForest']['feature_importance']
            top_indices = np.argsort(importance)[-10:][::-1]
            for idx in top_indices:
                report_lines.append(f"{self.feature_names[idx]}: {importance[idx]:.4f}")
        
        # Best model recommendation
        report_lines.append("\n\nMODEL RECOMMENDATION")
        report_lines.append("-" * 40)
        # Find best model for positive class precision
        best_model = max(train_results.keys(), 
                        key=lambda k: train_results[k]['precision_per_class'][2])
        best_precision = train_results[best_model]['precision_per_class'][2]
        
        report_lines.append(f"Best model for Positive class precision: {best_model}")
        report_lines.append(f"Positive class precision: {best_precision:.3f}")
        report_lines.append("")
        report_lines.append("TRADING INTERPRETATION:")
        report_lines.append(f"- Positive class (BUY signal) precision of {best_precision:.1%} means:")
        report_lines.append(f"  When the model predicts a stock will be in the top 33.3% of performers,")
        report_lines.append(f"  it is correct {best_precision:.1%} of the time.")
        report_lines.append(f"- This minimizes false positives (entering bad trades).")
        
        # Performance comparison to baselines
        report_lines.append("\n\nPERFORMANCE vs BASELINES")
        report_lines.append("-" * 40)
        random_accuracy = 1/3  # 33.3% for 3-class
        best_accuracy = train_results[best_model]['accuracy']
        improvement = (best_accuracy - random_accuracy) / random_accuracy * 100
        
        report_lines.append(f"Random baseline accuracy: {random_accuracy:.1%}")
        report_lines.append(f"Best model accuracy: {best_accuracy:.1%}")
        report_lines.append(f"Improvement over random: {improvement:.1f}%")
        
        report_lines.append("\n" + "=" * 80)
        
        # Save report
        with open(self.save_dir / 'multiclass_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📝 Comprehensive report saved to {self.save_dir}/multiclass_analysis_report.txt")

def main():
    """Run comprehensive multi-class analysis"""
    print("🎯 Multi-Class Comprehensive Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = MultiClassComprehensiveAnalyzer()
    
    # Load and prepare data
    X, y, setup_ids = analyzer.load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Data split: {len(X_train)} train, {len(X_test)} test")
    
    # Train models
    train_results = analyzer.train_models(X_train, y_train, X_test, y_test)
    
    # Cross-validation
    cv_results = analyzer.cross_validate_models(X, y)
    
    # Create visualizations
    analyzer.create_visualizations(train_results, cv_results, X_test, y_test)
    
    # Generate report
    analyzer.generate_report(train_results, cv_results, X, y)
    
    print(f"\n🎉 Comprehensive analysis complete!")
    print(f"📁 Results saved in: {analyzer.save_dir}/")
    
    return analyzer, train_results, cv_results

if __name__ == "__main__":
    analyzer, train_results, cv_results = main() 