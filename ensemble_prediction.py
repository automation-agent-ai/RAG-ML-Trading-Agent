#!/usr/bin/env python3
"""
Ensemble Prediction Tool

This script combines predictions from multiple domains to create an ensemble prediction:
1. Loads predictions from all domains
2. Combines them using various ensemble methods (majority vote, weighted average)
3. Evaluates the ensemble predictions against actual labels

Usage:
    python ensemble_prediction.py --method weighted --output ensemble_predictions.csv
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
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

from core.label_converter import LabelConverter, outperformance_to_class_int, PerformanceClass

class EnsemblePrediction:
    """Ensemble prediction across multiple domains"""
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        domains: List[str] = ["news", "fundamentals", "analyst_recommendations", "userposts"],
        label_converter: Optional[LabelConverter] = None
    ):
        self.db_path = db_path
        self.domains = domains
        self.label_converter = label_converter or LabelConverter()
        self.conn = duckdb.connect(db_path)
    
    def load_domain_predictions(self, setup_ids: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load predictions from all domains
        
        Args:
            setup_ids: Optional list of setup IDs to filter by
            
        Returns:
            Dictionary with domain names as keys and prediction DataFrames as values
        """
        domain_predictions = {}
        
        for domain in self.domains:
            logger.info(f"Loading predictions for domain: {domain}")
            
            # Build query
            query = f"""
            SELECT * FROM similarity_predictions
            WHERE domain = '{domain}'
            """
            
            # Add setup_id filter if provided
            if setup_ids:
                query += """
                AND setup_id IN (
                    SELECT UNNEST(?::VARCHAR[])
                )
                """
                # Execute query with parameters
                predictions_df = self.conn.execute(query, [setup_ids]).df()
            else:
                # Execute query without parameters
                predictions_df = self.conn.execute(query).df()
            
            if predictions_df.empty:
                logger.warning(f"No predictions found for domain: {domain}")
                continue
            
            # Add to dictionary
            domain_predictions[domain] = predictions_df
            logger.info(f"Loaded {len(predictions_df)} predictions for {domain}")
        
        return domain_predictions
    
    def load_actual_labels(self, setup_ids: List[str]) -> pd.DataFrame:
        """
        Load actual labels for evaluation
        
        Args:
            setup_ids: List of setup IDs to get labels for
            
        Returns:
            DataFrame with setup_ids and actual labels
        """
        query = """
        SELECT setup_id, outperformance_10d
        FROM labels
        WHERE setup_id IN (
            SELECT UNNEST(?::VARCHAR[])
        )
        """
        
        labels_df = self.conn.execute(query, [setup_ids]).df()
        logger.info(f"Loaded {len(labels_df)} actual labels")
        
        return labels_df
    
    def ensemble_majority_vote(self, domain_predictions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create ensemble prediction using majority vote
        
        Args:
            domain_predictions: Dictionary with domain predictions
            
        Returns:
            DataFrame with ensemble predictions
        """
        if not domain_predictions:
            logger.warning("No domain predictions available for ensemble")
            return pd.DataFrame()
        
        # Get common setup IDs across all domains
        common_setups = set()
        for i, (domain, df) in enumerate(domain_predictions.items()):
            setup_ids = set(df['setup_id'])
            if i == 0:
                common_setups = setup_ids
            else:
                common_setups = common_setups.intersection(setup_ids)
        
        if not common_setups:
            logger.warning("No common setup IDs across domains")
            return pd.DataFrame()
        
        logger.info(f"Found {len(common_setups)} common setup IDs across domains")
        
        # Create ensemble predictions
        ensemble_data = []
        
        for setup_id in sorted(common_setups):
            # Get class predictions from each domain
            domain_classes = {}
            domain_confidences = {}
            
            for domain, df in domain_predictions.items():
                setup_row = df[df['setup_id'] == setup_id]
                if not setup_row.empty:
                    # Convert outperformance to class
                    pred_outperformance = setup_row['predicted_outperformance'].iloc[0]
                    pred_class = self.label_converter.outperformance_to_class_int(pred_outperformance)
                    domain_classes[domain] = pred_class
                    domain_confidences[domain] = setup_row['confidence'].iloc[0]
            
            # Count votes for each class
            class_votes = {-1: 0, 0: 0, 1: 0}
            for cls in domain_classes.values():
                class_votes[cls] = class_votes.get(cls, 0) + 1
            
            # Find majority class
            max_votes = max(class_votes.values())
            majority_classes = [cls for cls, votes in class_votes.items() if votes == max_votes]
            
            # If tie, use highest confidence
            if len(majority_classes) > 1:
                # Find domain with highest confidence among tied classes
                max_confidence = -1
                selected_class = 0  # Default to neutral
                
                for domain, cls in domain_classes.items():
                    if cls in majority_classes and domain_confidences[domain] > max_confidence:
                        max_confidence = domain_confidences[domain]
                        selected_class = cls
            else:
                selected_class = majority_classes[0]
            
            # Convert class back to outperformance (use midpoint of range)
            if selected_class == 1:  # Positive
                ensemble_outperformance = 0.03  # Midpoint of positive range
            elif selected_class == -1:  # Negative
                ensemble_outperformance = -0.03  # Midpoint of negative range
            else:  # Neutral
                ensemble_outperformance = 0.0  # Midpoint of neutral range
            
            # Calculate overall confidence as average of domain confidences
            avg_confidence = sum(domain_confidences.values()) / len(domain_confidences)
            
            # Create ensemble prediction record
            ensemble_data.append({
                'setup_id': setup_id,
                'predicted_outperformance': ensemble_outperformance,
                'predicted_class': selected_class,
                'confidence': avg_confidence,
                'voting_domains': len(domain_classes),
                'positive_votes': class_votes.get(1, 0),
                'neutral_votes': class_votes.get(0, 0),
                'negative_votes': class_votes.get(-1, 0)
            })
        
        # Create DataFrame
        ensemble_df = pd.DataFrame(ensemble_data)
        logger.info(f"Created {len(ensemble_df)} ensemble predictions using majority vote")
        
        return ensemble_df
    
    def ensemble_weighted_average(self, domain_predictions: Dict[str, pd.DataFrame], 
                                 domain_weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Create ensemble prediction using weighted average
        
        Args:
            domain_predictions: Dictionary with domain predictions
            domain_weights: Optional dictionary with domain weights (default: equal weights)
            
        Returns:
            DataFrame with ensemble predictions
        """
        if not domain_predictions:
            logger.warning("No domain predictions available for ensemble")
            return pd.DataFrame()
        
        # Set default weights if not provided
        if domain_weights is None:
            domain_weights = {domain: 1.0 for domain in domain_predictions.keys()}
        
        # Normalize weights
        total_weight = sum(domain_weights.values())
        if total_weight == 0:
            logger.warning("Total domain weight is zero, using equal weights")
            domain_weights = {domain: 1.0 for domain in domain_predictions.keys()}
            total_weight = sum(domain_weights.values())
        
        normalized_weights = {domain: weight / total_weight 
                             for domain, weight in domain_weights.items()}
        
        # Get common setup IDs across all domains
        common_setups = set()
        for i, (domain, df) in enumerate(domain_predictions.items()):
            setup_ids = set(df['setup_id'])
            if i == 0:
                common_setups = setup_ids
            else:
                common_setups = common_setups.intersection(setup_ids)
        
        if not common_setups:
            logger.warning("No common setup IDs across domains")
            return pd.DataFrame()
        
        logger.info(f"Found {len(common_setups)} common setup IDs across domains")
        
        # Create ensemble predictions
        ensemble_data = []
        
        for setup_id in sorted(common_setups):
            # Get predictions from each domain
            domain_outperformances = {}
            domain_confidences = {}
            domain_positive_ratios = {}
            domain_negative_ratios = {}
            domain_neutral_ratios = {}
            
            for domain, df in domain_predictions.items():
                setup_row = df[df['setup_id'] == setup_id]
                if not setup_row.empty:
                    domain_outperformances[domain] = setup_row['predicted_outperformance'].iloc[0]
                    domain_confidences[domain] = setup_row['confidence'].iloc[0]
                    domain_positive_ratios[domain] = setup_row['positive_ratio'].iloc[0]
                    domain_negative_ratios[domain] = setup_row['negative_ratio'].iloc[0]
                    domain_neutral_ratios[domain] = setup_row['neutral_ratio'].iloc[0]
            
            # Calculate weighted average
            weighted_outperformance = sum(
                outperformance * normalized_weights.get(domain, 0) * domain_confidences.get(domain, 1.0)
                for domain, outperformance in domain_outperformances.items()
            ) / sum(
                normalized_weights.get(domain, 0) * domain_confidences.get(domain, 1.0)
                for domain in domain_outperformances.keys()
            )
            
            # Calculate weighted ratios
            weighted_positive_ratio = sum(
                ratio * normalized_weights.get(domain, 0)
                for domain, ratio in domain_positive_ratios.items()
            )
            
            weighted_negative_ratio = sum(
                ratio * normalized_weights.get(domain, 0)
                for domain, ratio in domain_negative_ratios.items()
            )
            
            weighted_neutral_ratio = sum(
                ratio * normalized_weights.get(domain, 0)
                for domain, ratio in domain_neutral_ratios.items()
            )
            
            # Calculate overall confidence as weighted average of domain confidences
            weighted_confidence = sum(
                confidence * normalized_weights.get(domain, 0)
                for domain, confidence in domain_confidences.items()
            )
            
            # Get predicted class
            predicted_class = self.label_converter.outperformance_to_class_int(weighted_outperformance)
            
            # Create ensemble prediction record
            ensemble_data.append({
                'setup_id': setup_id,
                'predicted_outperformance': weighted_outperformance,
                'predicted_class': predicted_class,
                'confidence': weighted_confidence,
                'domains_count': len(domain_outperformances),
                'positive_ratio': weighted_positive_ratio,
                'negative_ratio': weighted_negative_ratio,
                'neutral_ratio': weighted_neutral_ratio
            })
        
        # Create DataFrame
        ensemble_df = pd.DataFrame(ensemble_data)
        logger.info(f"Created {len(ensemble_df)} ensemble predictions using weighted average")
        
        return ensemble_df
    
    def evaluate_predictions(self, predictions_df: pd.DataFrame, labels_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate predictions against actual labels
        
        Args:
            predictions_df: DataFrame with predictions
            labels_df: DataFrame with actual labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Merge predictions with labels
        merged_df = pd.merge(
            predictions_df,
            labels_df,
            on='setup_id',
            how='inner'
        )
        
        if merged_df.empty:
            logger.warning("No matching predictions and labels for evaluation")
            return {}
        
        # Convert actual labels to classes
        merged_df['actual_class'] = merged_df['outperformance_10d'].apply(
            lambda x: self.label_converter.outperformance_to_class_int(x)
        )
        
        # Calculate metrics
        y_true = merged_df['actual_class']
        y_pred = merged_df['predicted_class']
        
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
        
        # Plot confusion matrix
        self.plot_confusion_matrix(merged_df, "ensemble")
        
        return metrics
    
    def plot_confusion_matrix(self, merged_df: pd.DataFrame, method: str, 
                             save_path: Optional[str] = None) -> None:
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
        plt.title(f'Confusion Matrix - {method.capitalize()} Ensemble')
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved confusion matrix to {save_path}")
        else:
            # Create output directory if it doesn't exist
            output_dir = Path("evaluation_results")
            output_dir.mkdir(exist_ok=True)
            save_path = output_dir / f"confusion_matrix_ensemble_{method}.png"
            plt.savefig(save_path)
            logger.info(f"Saved confusion matrix to {save_path}")
    
    def run_ensemble(self, method: str = 'weighted', setup_ids: Optional[List[str]] = None,
                    domain_weights: Optional[Dict[str, float]] = None,
                    output_file: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Run ensemble prediction
        
        Args:
            method: Ensemble method ('majority' or 'weighted')
            setup_ids: Optional list of setup IDs to process
            domain_weights: Optional dictionary with domain weights
            output_file: Optional file to save ensemble predictions
            
        Returns:
            Tuple of (ensemble predictions DataFrame, evaluation metrics)
        """
        # Load domain predictions
        domain_predictions = self.load_domain_predictions(setup_ids)
        
        if not domain_predictions:
            logger.error("No domain predictions available")
            return pd.DataFrame(), {}
        
        # Create ensemble predictions
        if method == 'majority':
            ensemble_df = self.ensemble_majority_vote(domain_predictions)
        else:  # weighted
            ensemble_df = self.ensemble_weighted_average(domain_predictions, domain_weights)
        
        if ensemble_df.empty:
            logger.error("Failed to create ensemble predictions")
            return ensemble_df, {}
        
        # Load actual labels for evaluation
        labels_df = self.load_actual_labels(ensemble_df['setup_id'].tolist())
        
        # Evaluate predictions
        metrics = self.evaluate_predictions(ensemble_df, labels_df)
        
        # Print evaluation summary
        logger.info("\n" + "="*50)
        logger.info(f"ENSEMBLE PREDICTION SUMMARY ({method.upper()})")
        logger.info("="*50)
        logger.info(f"Total predictions: {len(ensemble_df)}")
        logger.info(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        logger.info(f"F1 Score (weighted): {metrics.get('f1_weighted', 0):.4f}")
        
        # Class-specific metrics
        for cls_name in ['positive', 'neutral', 'negative']:
            logger.info(f"{cls_name.capitalize()} class F1: {metrics.get(f'f1_{cls_name}', 0):.4f}")
        
        # Save to file if requested
        if output_file:
            # Create output directory if needed
            output_path = Path(output_file)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Save predictions
            ensemble_df.to_csv(output_file, index=False)
            logger.info(f"Saved ensemble predictions to {output_file}")
            
            # Save metrics
            metrics_file = output_path.with_name(f"{output_path.stem}_metrics.csv")
            pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
            logger.info(f"Saved evaluation metrics to {metrics_file}")
        
        return ensemble_df, metrics
    
    def cleanup(self):
        """Close connections"""
        if hasattr(self, 'conn'):
            self.conn.close()
        logger.info("Connections closed")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Create ensemble predictions across domains")
    parser.add_argument("--method", choices=['majority', 'weighted'], default='weighted',
                      help="Ensemble method (default: weighted)")
    parser.add_argument("--db-path", default="data/sentiment_system.duckdb",
                      help="Path to DuckDB database")
    parser.add_argument("--setup-list", help="File containing setup IDs to process")
    parser.add_argument("--output", help="Output file for ensemble predictions")
    parser.add_argument("--domains", nargs="+", 
                      default=["news", "fundamentals", "analyst_recommendations", "userposts"],
                      help="Domains to include in ensemble")
    parser.add_argument("--weights", nargs="+", type=float,
                      help="Weights for each domain (must match number of domains)")
    
    args = parser.parse_args()
    
    # Load setup IDs if provided
    setup_ids = None
    if args.setup_list:
        with open(args.setup_list, 'r') as f:
            setup_ids = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(setup_ids)} setup IDs from {args.setup_list}")
    
    # Create domain weights if provided
    domain_weights = None
    if args.weights:
        if len(args.weights) != len(args.domains):
            parser.error("Number of weights must match number of domains")
        domain_weights = dict(zip(args.domains, args.weights))
        logger.info(f"Using custom domain weights: {domain_weights}")
    
    # Initialize ensemble prediction
    ensemble = EnsemblePrediction(
        db_path=args.db_path,
        domains=args.domains
    )
    
    try:
        # Run ensemble prediction
        ensemble_df, metrics = ensemble.run_ensemble(
            method=args.method,
            setup_ids=setup_ids,
            domain_weights=domain_weights,
            output_file=args.output
        )
    finally:
        # Clean up
        ensemble.cleanup()

if __name__ == "__main__":
    main() 