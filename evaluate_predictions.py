#!/usr/bin/env python3
"""
Prediction Evaluation Tool

This script evaluates prediction performance by comparing:
1. Similarity-based predictions from the pipeline
2. Saved labels from the data splitting process

Usage:
    python evaluate_predictions.py --domain news
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path for proper imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.label_converter import LabelConverter, outperformance_to_class_int


class PredictionEvaluator:
    """Evaluates prediction performance"""
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        labels_dir: str = "data"
    ):
        self.db_path = db_path
        self.labels_dir = labels_dir
        self.label_converter = LabelConverter()
        
    def load_predictions(self, domain: str) -> pd.DataFrame:
        """Load predictions from the database"""
        try:
            conn = duckdb.connect(self.db_path)
            
            query = f"""
            SELECT setup_id, predicted_outperformance, confidence, 
                   positive_ratio, negative_ratio, neutral_ratio
            FROM similarity_predictions
            WHERE domain = '{domain}'
            """
            
            predictions_df = conn.execute(query).df()
            conn.close()
            
            if predictions_df.empty:
                logger.warning(f"No predictions found for domain: {domain}")
                return pd.DataFrame()
            
            logger.info(f"Loaded {len(predictions_df)} predictions for domain: {domain}")
            return predictions_df
            
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            return pd.DataFrame()
    
    def load_labels(self, domain: str) -> pd.DataFrame:
        """Load saved labels from CSV file"""
        labels_file = Path(self.labels_dir) / f"prediction_labels_{domain}.csv"
        
        if not labels_file.exists():
            logger.warning(f"Labels file not found: {labels_file}")
            return pd.DataFrame()
        
        try:
            labels_df = pd.read_csv(labels_file)
            logger.info(f"Loaded {len(labels_df)} labels from {labels_file}")
            return labels_df
            
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            return pd.DataFrame()
    
    def merge_predictions_with_labels(self, predictions_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
        """Merge predictions with labels"""
        if predictions_df.empty or labels_df.empty:
            return pd.DataFrame()
        
        # Merge on setup_id
        merged_df = pd.merge(
            predictions_df, 
            labels_df,
            on='setup_id',
            how='inner'
        )
        
        logger.info(f"Merged {len(merged_df)} predictions with labels")
        
        # Convert to class labels
        merged_df['predicted_class'] = merged_df['predicted_outperformance'].apply(
            lambda x: outperformance_to_class_int(x)
        )
        
        merged_df['actual_class'] = merged_df['outperformance_10d'].apply(
            lambda x: outperformance_to_class_int(x)
        )
        
        return merged_df
    
    def calculate_metrics(self, merged_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        if merged_df.empty:
            return {}
        
        y_true = merged_df['actual_class']
        y_pred = merged_df['predicted_class']
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Class-specific metrics
        for cls in [-1, 0, 1]:
            cls_name = {-1: 'negative', 0: 'neutral', 1: 'positive'}[cls]
            
            # Binary classification metrics for this class
            y_true_bin = (y_true == cls).astype(int)
            y_pred_bin = (y_pred == cls).astype(int)
            
            metrics[f'precision_{cls_name}'] = precision_score(y_true_bin, y_pred_bin)
            metrics[f'recall_{cls_name}'] = recall_score(y_true_bin, y_pred_bin)
            metrics[f'f1_{cls_name}'] = f1_score(y_true_bin, y_pred_bin)
        
        return metrics
    
    def plot_confusion_matrix(self, merged_df: pd.DataFrame, domain: str, save_path: Optional[str] = None) -> None:
        """Plot confusion matrix"""
        if merged_df.empty:
            return
        
        y_true = merged_df['actual_class']
        y_pred = merged_df['predicted_class']
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive']
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {domain.capitalize()} Domain')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved confusion matrix to {save_path}")
        else:
            plt.show()
    
    def evaluate_domain(self, domain: str) -> Dict[str, float]:
        """Evaluate predictions for a specific domain"""
        # Load predictions and labels
        predictions_df = self.load_predictions(domain)
        labels_df = self.load_labels(domain)
        
        if predictions_df.empty or labels_df.empty:
            logger.warning(f"Cannot evaluate domain {domain}: missing data")
            return {}
        
        # Merge predictions with labels
        merged_df = self.merge_predictions_with_labels(predictions_df, labels_df)
        
        if merged_df.empty:
            logger.warning(f"No matching predictions and labels for domain {domain}")
            return {}
        
        # Calculate metrics
        metrics = self.calculate_metrics(merged_df)
        
        # Plot confusion matrix
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)
        self.plot_confusion_matrix(
            merged_df, 
            domain, 
            save_path=str(output_dir / f"confusion_matrix_{domain}.png")
        )
        
        # Save detailed results
        merged_df.to_csv(output_dir / f"evaluation_details_{domain}.csv", index=False)
        
        return metrics
    
    def evaluate_all_domains(self) -> Dict[str, Dict[str, float]]:
        """Evaluate predictions for all domains"""
        domains = ["news", "fundamentals", "analyst_recommendations", "userposts"]
        results = {}
        
        for domain in domains:
            logger.info(f"Evaluating domain: {domain}")
            metrics = self.evaluate_domain(domain)
            
            if metrics:
                results[domain] = metrics
                
                # Print metrics
                logger.info(f"Metrics for {domain}:")
                for metric, value in metrics.items():
                    logger.info(f"  {metric}: {value:.4f}")
            
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Evaluate prediction performance')
    parser.add_argument('--domain', choices=['news', 'fundamentals', 'analyst', 'userposts', 'all'],
                       default='all', help='Domain to evaluate (default: all)')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--labels-dir', default='data',
                       help='Directory containing prediction labels CSV files')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = PredictionEvaluator(
        db_path=args.db_path,
        labels_dir=args.labels_dir
    )
    
    # Evaluate domains
    if args.domain == 'all':
        results = evaluator.evaluate_all_domains()
    else:
        # Map 'analyst' to 'analyst_recommendations'
        domain = 'analyst_recommendations' if args.domain == 'analyst' else args.domain
        metrics = evaluator.evaluate_domain(domain)
        results = {domain: metrics} if metrics else {}
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    
    if not results:
        logger.warning("No evaluation results available")
    else:
        for domain, metrics in results.items():
            logger.info(f"\nDomain: {domain}")
            logger.info(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
            logger.info(f"F1 Score (weighted): {metrics.get('f1_weighted', 0):.4f}")
            
            # Class-specific metrics
            for cls_name in ['positive', 'neutral', 'negative']:
                logger.info(f"{cls_name.capitalize()} class F1: {metrics.get(f'f1_{cls_name}', 0):.4f}")


if __name__ == "__main__":
    main() 